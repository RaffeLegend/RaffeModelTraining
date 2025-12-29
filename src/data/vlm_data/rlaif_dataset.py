# src/data/hf_rlaif_dataset.py
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import functional as TF
import torch
from typing import Dict, Any, List
from src.data.vlm_data.base_dataset import BaseMultimodalDataset


class HFRLAIFDataset(BaseMultimodalDataset):
    def __init__(self, dataset_name: str, split="train", image_size=(224, 224), processor=None):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.processor = processor
        self.max_length = 1028

        # debug info
        sample = self.dataset[0]
        print("[HFRLAIFDataset] keys:", sample.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取数据
        images = [x["image"].convert("RGB") for x in batch]
        questions = [x["question"] for x in batch]
        answers = [x["chosen"] for x in batch]
        
        # 1. 构建符合Qwen格式的messages
        messages_list = []
        for img, q in zip(images, questions):
            messages_list.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q}
                ]},
                # {"role": "assistant", "content": ""}  # 占位，用于生成
            ])
        
        # 2. 处理文本tokenization（包含特殊token）
        text_inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,  # 添加assistant开始token
            padding=True,
            max_length=self.max_length,
            return_dict=True,
            return_tensors='pt' # return dict and tensors are neccessery
        )

        # # 3. 处理图像
        # image_inputs = self.processor.image_processor(
        #     images,
        #     return_tensors='pt'
        # )
        # print(image_inputs)
        # model_inputs = self.processor(
        #     text=text_inputs,
        #     images=images,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors="pt"
        # )        
        # print(model_inputs)


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
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": text_inputs["pixel_values"],
            "image_grid_thw": text_inputs["image_grid_thw"],
            "labels": labels,
        }

    def info(self):
        return {
            "type": "hf_rlaif",
            "size": len(self),
            "image_size": self.image_size
        }