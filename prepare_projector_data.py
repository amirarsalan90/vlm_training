import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("./data/projector_training/blip_laion_cc_sbu_558k.json")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, eval_df = train_test_split(df, test_size=5000, random_state=42)

train_df.to_json('./data/projector_training/chat_train.json', orient='records', lines=True)
eval_df.to_json('./data/projector_training/chat_val_5K.json', orient='records', lines=True)