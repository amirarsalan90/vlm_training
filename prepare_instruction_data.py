import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def get_hf_format(conv_list):
    new_conversation_list = []
    for talk in conv_list:
        temp = {}
        temp["role"] = talk['from']
        temp['content'] = talk['value']
        new_conversation_list.append(temp)

    return new_conversation_list

def get_prefix(thepath):
    return thepath.split("/")[0]

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")


df = pd.read_json("./data/instruction_finetuning/llava_v1_5_mix665k.json")

df['hf_format_conversations'] = df['conversations'].apply(get_hf_format)

image_df = df[df['image'].notna()]
noimage_df = df[df['image'].isna()]

image_df['prefix'] = image_df['image'].apply(get_prefix)

#image_df = image_df[image_df['prefix'].isin(["coco","gqa","textvqa"])]

final_df = pd.concat([image_df, noimage_df], axis=0).reset_index(drop=True)

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df = final_df[['id', 'image', 'hf_format_conversations']]

train_df, eval_df = train_test_split(final_df, test_size=5000, random_state=42)

train_df.to_json('./data/instruction_finetuning/instruction_train_all.json', orient='records', lines=True)
eval_df.to_json('./data/instruction_finetuning/instruction_val_5K_all.json', orient='records', lines=True)