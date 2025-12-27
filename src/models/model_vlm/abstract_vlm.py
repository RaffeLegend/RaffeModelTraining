from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
import json
from pathlib import Path
import deepspeed
from torch.utils.data.distributed import DistributedSampler
import os


class BaseFineTuner(ABC):
    """
    Abstract base class for model fine-tuning framework.
    Subclasses must implement abstract methods to customize fine-tuning logic.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_deepspeed: bool = False,
        deepspeed_config: Optional[str] = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_path: Path to pretrained model
            output_dir: Directory to save fine-tuned model
            device: Training device
            use_deepspeed: Whether to use DeepSpeed
            deepspeed_config: Path to DeepSpeed config file
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.model_engine = None
        
        # Initialize distributed training
        if use_deepspeed:
            deepspeed.init_distributed()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.world_size = 1
        
    @abstractmethod
    def load_model(self) -> None:
        """Load pretrained model (must be implemented by subclass)"""
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> None:
        """Load tokenizer (must be implemented by subclass)"""
        pass
    
    @abstractmethod
    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        Prepare dataset (must be implemented by subclass).
        
        Args:
            data_path: Path to data file
            
        Returns:
            Dataset object
        """
        pass
    
    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process single batch (must be implemented by subclass).
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary of processed tensors
        """
        pass
    
    def setup_lora(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05
    ) -> None:
        """
        Configure LoRA fine-tuning.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            target_modules: List of target modules
            lora_dropout: Dropout rate
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA configuration completed:")
        self.model.print_trainable_parameters()
    
    def setup_optimizer(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ) -> None:
        """
        Configure optimizer.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def train(
        self,
        data_path: str,
        batch_size: int = 4,
        num_epochs: int = 3,
        save_steps: int = 100,
        eval_steps: int = 50
    ) -> None:
        """
        Main training loop.
        
        Args:
            data_path: Path to training data
            batch_size: Batch size per device
            num_epochs: Number of training epochs
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
        """
        # Prepare data
        dataset = self.prepare_dataset(data_path)
        
        # Use DistributedSampler for multi-GPU training
        if self.use_deepspeed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=self.collate_fn if hasattr(self, 'collate_fn') else None
            )
        
        # Initialize DeepSpeed
        if self.use_deepspeed:
            ds_config = self._get_deepspeed_config() if self.deepspeed_config is None else self.deepspeed_config
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=ds_config,
                model_parameters=self.model.parameters()
            )
        else:
            # Training mode
            self.model.train()
            self.model.to(self.device)
        
        global_step = 0
        
        for epoch in range(num_epochs):
            if self.use_deepspeed and hasattr(dataloader, 'sampler'):
                dataloader.sampler.set_epoch(epoch)
            
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Process batch data
                processed_batch = self.process_batch(batch)
                
                if self.use_deepspeed:
                    # DeepSpeed training step
                    outputs = self.model_engine(**processed_batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    # Standard training step
                    outputs = self.model(**processed_batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging (only on rank 0)
                if global_step % 10 == 0 and self.local_rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Step {global_step}, "
                          f"Loss: {loss.item():.4f}")
                
                # Save checkpoint (only on rank 0)
                if global_step % save_steps == 0 and self.local_rank == 0:
                    self.save_checkpoint(global_step)
                
                # Evaluate (if subclass implements evaluate method)
                if global_step % eval_steps == 0 and hasattr(self, 'evaluate'):
                    self.evaluate()
                    if self.use_deepspeed:
                        self.model_engine.train()
                    else:
                        self.model.train()
            
            avg_loss = epoch_loss / len(dataloader)
            if self.local_rank == 0:
                print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")
        
        # Save final model (only on rank 0)
        if self.local_rank == 0:
            self.save_model()
    
    def _get_deepspeed_config(self) -> Dict:
        """
        Get default DeepSpeed configuration.
        Can be overridden by subclass for custom configs.
        """
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5,
                    "warmup_num_steps": 100
                }
            }
        }
    
    def save_checkpoint(self, step: int) -> None:
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        if self.use_deepspeed:
            self.model_engine.save_checkpoint(str(checkpoint_dir))
        else:
            self.model.save_pretrained(checkpoint_dir)
            
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to: {checkpoint_dir}")
    
    def save_model(self) -> None:
        """Save final model"""
        if self.use_deepspeed:
            self.model_engine.save_checkpoint(str(self.output_dir))
        else:
            self.model.save_pretrained(self.output_dir)
            
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to: {self.output_dir}")
    
    @abstractmethod
    def inference(self, input_data: Any) -> str:
        """
        Inference method (must be implemented by subclass).
        
        Args:
            input_data: Input data
            
        Returns:
            Model output
        """
        pass
    
    def load_finetuned_model(self) -> None:
        """Load fine-tuned model"""
        self.load_model()
        self.load_tokenizer()
        print(f"Fine-tuned model loaded from {self.output_dir}")