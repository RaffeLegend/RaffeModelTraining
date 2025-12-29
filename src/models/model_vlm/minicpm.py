from PIL import Image
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
import json

from src.models.model_vlm.abstract_vlm import BaseFineTuner
from src.data.vlm_data.rlaif_dataset import HFRLAIFDataset

import os, psutil, time
def mem(tag):
    rss = psutil.Process(os.getpid()).memory_info().rss/1024**3
    print(f"[MEM] {tag}: RSS={rss:.2f} GB", flush=True)


class MiniCPMFineTuner(BaseFineTuner):
    """MiniCPM model fine-tuning implementation"""

    def load_model(self):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.float16,
            device_map="cuda", # key setting
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def prepare_dataset(self, data_path: str) -> HFRLAIFDataset:
        return HFRLAIFDataset(
            dataset_name=data_path,
            split="train",
            image_size=(224, 224)
        )

    def process_batch(self, batch):
        images = [x["image"].convert("RGB") for x in batch]
        questions = [x["question"] for x in batch]
        answers = [x["chosen"] for x in batch]
        
        print(f"=== Processing {len(images)} images ===")
        
        # 方法：统一所有图像到一个固定的切片数量
        # 1. 确保所有图像有相同数量的切片
        all_slice_counts = []
        for img in images:
            processed = self.processor.image_processor(img, return_tensors="pt")
            slice_count = len(processed["image_sizes"][0])
            all_slice_counts.append(slice_count)
        
        print(f"Slice counts: {all_slice_counts}")
        
        # 找到最大切片数
        max_slices = max(all_slice_counts)
        print(f"Max slices: {max_slices}")
        
        # 2. 构建提示 - 使用最大切片数
        prompts = []
        for q in questions:
            # 为最大切片数创建标记
            image_tags = "".join([f"<image>./</image>" for _ in range(max_slices)])
            prompt = f"{image_tags}\nUser: {q}\nAssistant:"
            prompts.append(prompt)
        
        print(f"First prompt (truncated): {prompts[0][:100]}...")
        
        # 3. 关键：设置 max_slice_nums 参数
        try:
            prompt_inputs = self.processor(
                text=prompts,
                images=images,
                padding=True,  # 这很重要，确保长度一致
                return_tensors="pt",
                max_slice_nums=max_slices,  # 告诉处理器期望的最大切片数
                truncation=True,
                max_length=2048
            )
            print("Batch processing SUCCESS!")
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            
            # 备用方案：逐样本处理并手动填充
            print("Using sample-by-sample with padding...")
            all_prompt_inputs = []
            
            for i, (img, prompt) in enumerate(zip(images, prompts)):
                try:
                    single_input = self.processor(
                        text=prompt,
                        images=img,
                        padding=True,  # 单个样本也填充
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048
                    )
                    all_prompt_inputs.append(single_input)
                    print(f"Sample {i}: {single_input['input_ids'].shape[1]} tokens")
                except Exception as e2:
                    print(f"Sample {i} failed: {e2}")
                    raise
            
            # 手动堆叠（现在应该长度相同了）
            prompt_inputs = {
                "input_ids": torch.cat([x["input_ids"] for x in all_prompt_inputs], dim=0),
                "attention_mask": torch.cat([x["attention_mask"] for x in all_prompt_inputs], dim=0),
                "pixel_values": torch.cat([x["pixel_values"] for x in all_prompt_inputs], dim=0),
            }
            print(f"Stacked shapes: {prompt_inputs['input_ids'].shape}")
        
        # 4. 处理答案
        answer_inputs = self.tokenizer(
            answers,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        
        # 5. 拼接
        input_ids = torch.cat([prompt_inputs["input_ids"], answer_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([
            prompt_inputs["attention_mask"],
            torch.ones_like(answer_inputs["input_ids"])
        ], dim=1)
        
        # 6. 构建标签
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        
        print(f"Final shapes: input_ids={input_ids.shape}, labels={labels.shape}")
        
        return {
            "input_ids": input_ids.cuda(),
            "attention_mask": attention_mask.cuda(),
            "pixel_values": prompt_inputs["pixel_values"].cuda(),
            "labels": labels.cuda(),
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