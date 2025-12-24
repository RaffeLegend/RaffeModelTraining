from PIL import Image
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
import json

from src.models.model_vlm.abstract_vlm import BaseFineTuner


class MiniCPMFineTuner(BaseFineTuner):
    """MiniCPM model fine-tuning implementation"""
    
    def load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        import pandas as pd
        from pathlib import Path
        
        class MiniCPMDataset(Dataset):
            def __init__(self, data_source):
                self.data = []
                
                # Case 1: JSON file
                if isinstance(data_source, str) and data_source.endswith('.json'):
                    with open(data_source, 'r', encoding='utf-8') as f:
                        self.data = json.load(f)
                
                # Case 2: Single parquet file
                elif isinstance(data_source, str) and data_source.endswith('.parquet'):
                    df = pd.read_parquet(data_source)
                    self.data = df.to_dict('records')
                
                # Case 3: Directory with multiple parquet files
                elif isinstance(data_source, str) and Path(data_source).is_dir():
                    parquet_files = list(Path(data_source).glob('*.parquet'))
                    if parquet_files:
                        dfs = [pd.read_parquet(f) for f in parquet_files]
                        combined_df = pd.concat(dfs, ignore_index=True)
                        self.data = combined_df.to_dict('records')
                    else:
                        raise ValueError(f"No parquet files found in {data_source}")
                
                # Case 4: List of parquet files
                elif isinstance(data_source, list):
                    dfs = [pd.read_parquet(f) for f in data_source]
                    combined_df = pd.concat(dfs, ignore_index=True)
                    self.data = combined_df.to_dict('records')
                
                # Case 5: HuggingFace dataset name
                else:
                    from datasets import load_dataset
                    dataset = load_dataset(data_source, split='train')
                    self.data = [dataset[i] for i in range(len(dataset))]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
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
        model_path="openbmb/MiniCPM-V-2_6",
        output_dir="./output/minicpm_finetuned",
        use_deepspeed=True,  # Enable DeepSpeed
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