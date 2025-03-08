from datasets import load_dataset
from langdetect import detect, LangDetectException
import os
import pandas as pd


def lord_dataset():
    dataset = load_dataset("amay01/llm-sgd-dst8-training-data", split="train", cache_dir="D:/Download/DATA")
    sampled_data = dataset.shuffle(seed=42).select(range(1000))
    df = pd.DataFrame(sampled_data)
    output_path = os.path.join(os.path.dirname(__file__), "sgd_1000.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")


def washing():
    df = pd.read_csv('sgd_1000.csv')
    print(df['input'].values)
    chat_data = pd.DataFrame(columns=['system', 'user', 'intent'])
    for index, row in df.iterrows():
        intent = row['output']
        chat = row['input']
        array = chat.split('user: ')
        user_message = array[1]
        system_message = array[0].split('system: ')[1].replace('\n', '')
        intent_message = intent
        new_row = pd.Series([system_message, user_message, intent_message], index=['system', 'user', 'intent'])
        chat_data = pd.concat([chat_data, pd.DataFrame([new_row])], ignore_index=True)
    chat_data.index.name = 'id'
    chat_data.to_csv("chats_data.csv", encoding="utf-8")


lord_dataset()
washing()
