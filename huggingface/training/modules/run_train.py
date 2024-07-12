import pandas as pd
from transformers import TrainingArguments
from transformers import Trainer

from model import get_processor, get_model, ImageTextInstructionFollowingDataset, MyCustomDataCollator
from utils import freeze_network, count_trainable_parameters


processor = get_processor(
    image_processor_id="openai/clip-vit-large-patch14", 
    tokenizer_id="microsoft/phi-1_5", 
    image_token="<image>"
    )

model = get_model(vision_tower_id="openai/clip-vit-large-patch14", 
                  language_model_id="microsoft/phi-1_5", 
                  tokenizer=processor.tokenizer, 
                  image_token="<image>"
                  )

train_df = pd.read_json("/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/chat_train.json", lines=True)

train_dataset = ImageTextInstructionFollowingDataset(data=train_df, 
                                                     image_folder_path="/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/images"
                                                     )

eval_df = pd.read_json("/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/chat_val_5K.json", lines=True)
eval_dataset = ImageTextInstructionFollowingDataset(data=eval_df, 
                                                    image_folder_path="/home/arsalan/Desktop/multimodal_LLM-master/multimodal_LLM-master/data/images"
                                                    )

data_collator = MyCustomDataCollator(processor=processor)


training_args = TrainingArguments(
    #report_to = 'wandb',
    output_dir="./outputs/phi_adaptor-test3",
    remove_unused_columns=False,
    save_strategy="steps",
    save_steps=100,
    learning_rate=1e-5,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=4,
    #bf16=True,
    #bf16_full_eval=True,
    warmup_steps=0,
    lr_scheduler_type="linear",
    lr_scheduler_kwargs={},
    max_steps=3000,
    dataloader_num_workers=14,
    logging_steps=10,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=1000,
    label_names=["labels"],
    )



trainable_params = count_trainable_parameters(model)
print(f"Number of trainable parameters: {trainable_params}")

trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

trainer.train()