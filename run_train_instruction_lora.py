import requests
import os

import torch
import pandas as pd
import wandb
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig

from modules.model import get_model, get_processor
from modules.model import get_processor, get_model, ImageTextInstructionFollowingDataset2, MyCustomDataCollator2


#os.environ["WANDB_PROJECT"]="vlm"
#logging.set_verbosity_error()
#wandb.login()


processor = get_processor(
    image_processor_id="openai/clip-vit-large-patch14", 
    tokenizer_id="microsoft/phi-1_5", 
    image_token="<image>"
    )

checkpoint_path = "./training/outputs/phi_adaptor_hf/checkpoint-4000/"  # or the specific checkpoint you want to load

model = LlavaForConditionalGeneration.from_pretrained(checkpoint_path)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

PHI_CHAT_TEMPLATE = processor.tokenizer.default_chat_template
processor.tokenizer.chat_template = PHI_CHAT_TEMPLATE
    

data_collator = MyCustomDataCollator2(processor=processor)

train_df = pd.read_json("/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/instruction_finetune/instruction_train_all.json", lines=True)
train_dataset = ImageTextInstructionFollowingDataset2(data=train_df, 
                                                     image_folder_path="/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/instruction_finetune"
                                                     )

eval_df = pd.read_json("/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/instruction_finetune/instruction_val_5K_all.json", lines=True)
eval_dataset = ImageTextInstructionFollowingDataset2(data=eval_df, 
                                                     image_folder_path="/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/instruction_finetune"
                                                     )

training_args = TrainingArguments(
    output_dir="./outputs/phi_instruction",
    report_to="wandb",
    learning_rate=1.4e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=5,
    num_train_epochs=1,
    push_to_hub=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    fp16=True,
    bf16=False
)


lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules="all-linear"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",  # need a dummy field
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    dataset_kwargs={},
)

trainer.train()
trainer.save_model("./outputs/phi_instruction_final")