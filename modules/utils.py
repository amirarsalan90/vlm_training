import torch
import torch.nn as nn
from PIL import Image

class ProjectorTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_folder_path=None):
        self.data = data
        self.image_folder_path = image_folder_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversation = self.data["conversations"][index]
        image_path = self.data["image"][index]
        if self.image_folder_path is not None:
            image_path = f"{self.image_folder_path}/{image_path}"
        img = Image.open(image_path)
        instruction = conversation[0]["value"]
        answer = conversation[1]["value"]

        instruction_answer = instruction + "\n\nAnswer:" + answer
        return instruction_answer, img


class ProjectorTrainingDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        instruction_answers = [item[0] for item in examples]
        images = [item[1] for item in examples]
        
        tokenizer_outputs = self.processor.tokenizer(
            instruction_answers, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = tokenizer_outputs['input_ids']
        attention_mask = tokenizer_outputs['attention_mask']

        pixel_values = self.processor.image_processor(
            images=images, return_tensors="pt"
        )["pixel_values"]

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': input_ids
        }
        return batch


class InstructionFineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_folder_path=None):
        self.data = data
        self.image_folder_path = image_folder_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversation = self.data["hf_format_conversations"][index]
        image_path = self.data["image"][index]
        img = None
        if self.image_folder_path is not None:
            if image_path:
                image_path = f"{self.image_folder_path}/{image_path}"
                try:
                    img = Image.open(image_path)
                except FileNotFoundError:
                    print(f"Image not found at {image_path}")
        return conversation, img

class InstructionFineTuningDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        conversations = [item[0] for item in examples]
        images = [item[1] for item in examples]
        conversations_formatted = [self.processor.tokenizer.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
        tokenizer_outputs = self.processor.tokenizer(
            conversations_formatted, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = tokenizer_outputs['input_ids']
        attention_mask = tokenizer_outputs['attention_mask']
        
        if any(img is not None for img in images):
            non_none_images = [img for img in images if img is not None]
            pixel_values = self.processor.image_processor(
                images=non_none_images, 
                return_tensors="pt"
            )["pixel_values"]
        else:
            pixel_values = [None for i in range(len(examples))]

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': input_ids
        }
        return batch


def freeze_network(model):
    for param in model.language_model.parameters():
        param.requires_grad = False
    
    for param in model.vision_tower.parameters():
        param.requires_grad = False


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)