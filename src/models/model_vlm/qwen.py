from PIL import Image
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
import json

from src.models.model_vlm.abstract_vlm import BaseFineTuner
from src.data.vlm_data.rlaif_dataset import HFRLAIFDataset

import os, psutil, time

def mem(tag):
    rss = psutil.Process(os.getpid()).memory_info().rss/1024**3
    print(f"[MEM] {tag}: RSS={rss:.2f} GB", flush=True)


class QwenVLFineTuner(BaseFineTuner):
    """Qwen-VL model fine-tuning implementation"""

    def load_model(self):
        # Qwen-VL 使用标准的 CausalLM 模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="cuda",  # key setting
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
    
    def load_tokenizer(self):
        pass
    
    def load_processor(self):
        # Qwen-VL 使用 AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        
        # 对于 Qwen-VL，processor 包含了 tokenizer 和 image_processor
        # 但我们可以直接使用 tokenizer 进行文本处理
    
    def prepare_dataset(self, data_path: str) -> HFRLAIFDataset:
        return HFRLAIFDataset(
            dataset_name=data_path,
            split="train",
            image_size=(224, 224),  # Qwen-VL 常用尺寸
            processor=self.processor
        )
    
    def inference(self, input_data: Dict[str, Any]) -> str:
        """推理 - 完全使用官方示例的格式"""
        self.model.eval()
        image = Image.open(input_data['image']).convert('RGB')
        question = input_data['question']
        
        # 构建官方格式的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # 使用 apply_chat_template（官方推荐方式）
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        # 解码回复
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response


# ============= Usage Example =============

if __name__ == "__main__":
    # 选择 Qwen-VL 模型
    # 可选模型:
    # - "Qwen/Qwen2-VL-2B-Instruct" (轻量版)
    # - "Qwen/Qwen2-VL-7B-Instruct" (标准版)
    # - "Qwen/Qwen-VL-Chat" (对话优化版)
    
    model_choice = "Qwen/Qwen3-VL-4B-Instruct"
    
    # Initialize fine-tuner
    finetuner = QwenVLFineTuner(
        model_path=model_choice,
        output_dir="./output/qwenvl_finetuned",
        use_deepspeed=False,  # 如果需要 DeepSpeed 可以启用
        deepspeed_config="src/options/config/deepspeed_config.json"
    )
    
    # Load model, tokenizer and processor
    finetuner.load_model()
    finetuner.load_tokenizer()
    finetuner.load_processor()
    
    # Configure LoRA
    finetuner.setup_lora(r=8, lora_alpha=16)
    finetuner.setup_optimizer()
    
    # Start training
    finetuner.train(
        data_path="openbmb/RLAIF-V-Dataset",
        batch_size=1,  # Per-device batch size
        num_epochs=3,
        save_steps=100
    )
    
    # Inference
    result = finetuner.inference({
        'image': 'test.jpg',
        'question': 'What is in this image?'
    })
    print(f"Inference result: {result}")