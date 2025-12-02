import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

class MultiModalDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_processor, max_len=2048):
        self.data = [json.loads(l) for l in open(json_path)]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = item["image"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        msg = item["messages"]

        text = ""
        for m in msg:
            if m["role"] == "system":
                text += f"<s>[SYSTEM]\n{m['content']}\n</s>"
            elif m["role"] == "user":
                text += f"<s>[USER]\n{m['content']}\n</s>"
            elif m["role"] == "assistant":
                text += f"<s>[ASSISTANT]\n{m['content']}\n</s>"

        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
            "pixel_values": image_tensor
        }

class VisionProjector(torch.nn.Module):
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.proj = torch.nn.Linear(vision_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        images = torch.stack([b["pixel_values"] for b in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "labels": input_ids.clone()
        }

def main():
    model_name = "meta-llama/Meta-Llama-3-8B"  
    train_json = "./data/train.jsonl"
    val_json   = "./data/val.jsonl"
    output_dir = "./checkpoints/llama8b_multimodal"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)
    vision_encoder_name = "openai/clip-vit-large-patch14"
    vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
    image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_name)

    for p in vision_encoder.parameters():
        p.requires_grad = False

    vision_hidden = vision_encoder.config.hidden_size
    llm_hidden = model.config.hidden_size
    projector = VisionProjector(vision_hidden, llm_hidden)
    projector.train()

    model.vision_encoder = vision_encoder
    model.vision_projector = projector
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    train_dataset = MultiModalDataset(train_json, tokenizer, image_processor)
    val_dataset = MultiModalDataset(val_json, tokenizer, image_processor)
    collator = DataCollator(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=True,
        fp16=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    torch.save(projector.state_dict(), os.path.join(output_dir, "vision_projector.pt"))

    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
