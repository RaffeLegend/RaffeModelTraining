from PIL import Image
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
import json

from src.models.model_vlm.abstract_vlm import BaseFineTuner

import os, psutil, time
def mem(tag):
    rss = psutil.Process(os.getpid()).memory_info().rss/1024**3
    print(f"[MEM] {tag}: RSS={rss:.2f} GB", flush=True)


class MiniCPMFineTuner(BaseFineTuner):
    """MiniCPM model fine-tuning implementation"""
    
    def load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float16,
            device_map="cuda", # key setting
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        import json
        import os
        import psutil
        from pathlib import Path
        from torch.utils.data import Dataset

        print(
            "RSS(GB) before dataset init:",
            psutil.Process(os.getpid()).memory_info().rss / 1e9,
            flush=True
        )

        class MiniCPMDataset(Dataset):
            def __init__(self, data_source):
                self.source_type = None
                self.data = None
                self.files = None

                # ---------- Case 1: JSON file ----------
                if isinstance(data_source, str) and data_source.endswith(".json"):
                    self.source_type = "json"
                    with open(data_source, "r", encoding="utf-8") as f:
                        # ⚠️ JSON 本身就没法 mmap，小数据 OK，大数据不建议
                        self.data = json.load(f)

                # ---------- Case 2: Single parquet file ----------
                elif isinstance(data_source, str) and data_source.endswith(".parquet"):
                    self.source_type = "hf_parquet"
                    from datasets import load_dataset
                    self.data = load_dataset(
                        "parquet",
                        data_files=data_source,
                        split="train"
                    )

                # ---------- Case 3: Directory of parquet files ----------
                elif isinstance(data_source, str) and Path(data_source).is_dir():
                    parquet_files = sorted(Path(data_source).glob("*.parquet"))
                    if not parquet_files:
                        raise ValueError(f"No parquet files found in {data_source}")

                    self.source_type = "hf_parquet"
                    from datasets import load_dataset
                    self.data = load_dataset(
                        "parquet",
                        data_files=[str(p) for p in parquet_files],
                        split="train"
                    )

                # ---------- Case 4: List of parquet files ----------
                elif isinstance(data_source, (list, tuple)):
                    self.source_type = "hf_parquet"
                    from datasets import load_dataset
                    self.data = load_dataset(
                        "parquet",
                        data_files=[str(p) for p in data_source],
                        split="train"
                    )

                # ---------- Case 5: HuggingFace dataset name ----------
                else:
                    self.source_type = "hf_dataset"
                    from datasets import load_dataset
                    self.data = load_dataset(data_source, split="train")

                print(
                    f"[Dataset] type={self.source_type}, size={len(self)}",
                    flush=True
                )
                print(
                    "RSS(GB) after dataset init:",
                    psutil.Process(os.getpid()).memory_info().rss / 1e9,
                    flush=True
                )

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                # 这里仍然是 lazy access，不会一次性加载
                return self.data[idx]

        return MiniCPMDataset(data_path)
    
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = [Image.open(item['image']).convert('RGB') for item in batch]
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        
        # Build message format
        msgs_list = [[{'role': 'user', 'content': q}] for q in questions]
        
        # Process inputs
        inputs = self.model.process_messages(
            msgs_list[0],
            images,
            tokenizer=self.tokenizer
        )
        
        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'pixel_values': inputs.get('pixel_values').to(self.device),
            'labels': self.tokenizer.encode(answers[0], return_tensors='pt').to(self.device)
        }
    
    def inference(self, input_data: Dict[str, Any]) -> str:
        self.model.eval()
        image = Image.open(input_data['image']).convert('RGB')
        question = input_data['question']
        
        msgs = [{'role': 'user', 'content': question}]
        response = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return response


# ============= Usage Example =============

if __name__ == "__main__":
    # Initialize fine-tuner with DeepSpeed
    finetuner = MiniCPMFineTuner(
        model_path="openbmb/MiniCPM-V-4",
        output_dir="./output/minicpm_finetuned",
        use_deepspeed=False,  # Enable DeepSpeed
        deepspeed_config="src/options/config/deepspeed_config.json"  # Optional: custom config
    )
    
    # Load model and tokenizer
    finetuner.load_model()
    finetuner.load_tokenizer()
    
    # Configure LoRA
    finetuner.setup_lora(r=8, lora_alpha=32)
    
    # No need to setup optimizer when using DeepSpeed
    # (optimizer is configured in DeepSpeed config)
    
    # Start training
    finetuner.train(
        data_path="openbmb/RLAIF-V-Dataset",
        batch_size=2,  # Per-device batch size
        num_epochs=3,
        save_steps=100
    )
    
    # Inference
    result = finetuner.inference({
        'image': 'test.jpg',
        'question': 'What is in this image?'
    })