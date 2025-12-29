# src/data/hf_rlaif_dataset.py
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import functional as TF
import torch
from typing import Dict, Any, List
from src.data.vlm_data.base_dataset import BaseMultimodalDataset


class HFRLAIFDataset(BaseMultimodalDataset):
    def __init__(self, dataset_name: str, split="train", image_size=(224, 224), processor=None, device=None):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.processor = processor
        self.max_length = 1028
        self.device = device

        # debug info
        sample = self.dataset[0]
        print("[HFRLAIFDataset] keys:", sample.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取数据
        images = [x["image"].convert("RGB").resize(self.image_size) for x in batch]
        questions = [x["question"] for x in batch]
        answers = [x["chosen"] for x in batch]
        
        # 1. 构建符合Qwen格式的messages
        messages_list = []
        messages = []
        for img, q in zip(images, questions):
            # messages.append([{"role": "user", "content": q}])
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q}
                ]},
                # {"role": "assistant", "content": ""}  # 占位，用于生成
            ]
        
            # 2. 处理文本tokenization（包含特殊token）
            text_input = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 添加assistant开始token
                padding=True,
                max_length=self.max_length
                # return_dict=True,
                # return_tensors='pt' # return dict and tensors are neccessary
            )
            print(text_input)

            messages_list.append(text_input)

        # Process the text and images using the processor
        text_inputs = self.processor(
            text=messages_list,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        print(text_inputs["pixel_values"].shape)

        # 4. 处理答案tokenization
        answer_inputs = self.processor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors='pt'
        )
        
        # 5. 拼接input_ids
        input_ids = torch.cat([
            text_inputs["input_ids"],
            answer_inputs["input_ids"]
        ], dim=1)
        
        # 6. 构建attention_mask
        attention_mask = torch.cat([
            text_inputs["attention_mask"],
            answer_inputs["attention_mask"]
        ], dim=1)
        
        # 7. 创建labels（mask prompt部分）
        labels = input_ids.clone()
        labels[:, :text_inputs["input_ids"].shape[1]] = -100

        # 6. 确保pixel_values格式正确
        pixel_values = text_inputs["pixel_values"]
        if pixel_values.dim() == 2:
            # 如果是[N, 1536]，转换为[batch, N, 1536]
            batch_size = len(batch)
            N = pixel_values.shape[0] // batch_size
            D = pixel_values.shape[1]
            pixel_values = pixel_values.reshape(batch_size, N, D)
            print(f"重塑pixel_values: {pixel_values.shape}")
        
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": text_inputs["pixel_values"].to(self.device),
            "image_grid_thw": text_inputs["image_grid_thw"].to(self.device),
            "labels": labels.to(self.device),
        }

    def info(self):
        return {
            "type": "hf_rlaif",
            "size": len(self),
            "image_size": self.image_size
        }