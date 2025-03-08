import json
import os

from GPT.LLMClient import LLMClient
import pandas as pd

GENERATION = """
You are a professional consultant who is proficient in psychology and personality analysis. You need to conduct an in-depth analysis of the provided conversation content based on the "Big Five Personality Model" and give a score and detailed explanation for each dimension. Please follow the steps below to perform the task:
Understand the Big Five Personality Model and confirm your understanding of the five dimensions of the Big Five Personality and their sub-traits:
Openness: imagination, curiosity, artistic sensitivity, willingness to try new things.
Conscientiousness: self-discipline, goal orientation, organizational ability, sense of responsibility.
Extraversion: social activity, enthusiasm, self-confidence level, energy source (external/internal).
Agreeableness: cooperative tendency, empathy, trust, altruistic behavior.
Neuroticism: emotional stability, anxiety tendency, stress coping style, vulnerability.
If there are vague concepts, please calibrate yourself through authoritative psychological materials first.
"""
FORMULA = """
You must return a string to me in the following json format without other messages. Keep two decimal places. 
The scoring order is openness, responsibility, extraversion, agreeableness, neuroticism
{
"system": {
"dialogue": "Conversation content",
"bigfive_scores": [0.00, 0.00, 0.00, 0.00, 0.00] 
},
"user": {
"dialogue": "Conversation content",
"bigfive_scores": [0.00, 0.00, 0.00, 0.00, 0.00] 
}
}
"""


def extract_message(json_string):
    # 解析JSON字符串
    json_data = json.loads(json_string)

    # 提取需要的四个值
    system_context = json_data["system"]["dialogue"]
    user_context = json_data["user"]["dialogue"]
    system_bf = json_data["system"]["bigfive_scores"]
    user_bf = json_data["user"]["bigfive_scores"]

    # 返回包含这四个值的列表
    return [system_context, user_context, system_bf, user_bf]


def generation(llm, df):
    csv_path = "results/generation_data.csv"

    # 创建结果目录（如果不存在）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 初始化或续接处理进度
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        start_id = existing_df['id'].max() + 1
    else:
        # 创建带列名的空文件
        pd.DataFrame(columns=['id', 'system', 'user', 'intent',
                              'new_system', 'new_user', 'system_bf', 'user_bf']
                     ).to_csv(csv_path, index=False)
        start_id = 0

    for index in range(start_id, len(df)):
        row = df.iloc[index]
        system = row['system']
        user = row['user']
        intent = row['intent']

        TASK = f"""
        Next, I will give you a conversation intent. Please generate a conversation between the system and the user 
        according to the conversation intent, one sentence each. The new conversation needs to be able to show the user's 
        bigfive characteristics. The intent as follow:{intent}
        """

        prompt = GENERATION + TASK + FORMULA
        response = llm.get_response(prompt)
        context_list = extract_message(response)

        # 构造新数据行
        new_row = {
            'id': index,
            'system': system,
            'user': user,
            'intent': intent,
            'new_system': context_list[0],
            'new_user': context_list[1],
            'system_bf': context_list[2],
            'user_bf': context_list[3]
        }

        pd.DataFrame([new_row]).to_csv(csv_path,
                                       mode='a',
                                       header=False,
                                       index=False)

        print(f"Processed id {index}: {context_list}")

    return


def transformation(llm, df):
    csv_path = "results/transformation_data.csv"

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        start_id = existing_df['id'].max() + 1
    else:
        # 创建带列名的空文件
        pd.DataFrame(columns=['id', 'system', 'user', 'intent',
                              'new_system', 'new_user', 'system_bf', 'user_bf']
                     ).to_csv(csv_path, index=False)
        start_id = 0

    for index in range(start_id, len(df)):
        row = df.iloc[index]
        system = row['system']
        user = row['user']
        intent = row['intent']

        TASK = f"""Next, I will give you a conversation. Please rewrite or extend the conversation according to the 
        original meaning. The new conversation needs to be able to show the user's bigfive characteristics. The 
        conversation is as follows:system:{system},user:{user}"""

        prompt = GENERATION + TASK + FORMULA
        response = llm.get_response(prompt)
        print(response)
        context_list = extract_message(response)

        # 构造新数据行
        new_row = {
            'id': index,
            'system': system,
            'user': user,
            'intent': intent,
            'new_system': context_list[0],
            'new_user': context_list[1],
            'system_bf': context_list[2],
            'user_bf': context_list[3]
        }

        pd.DataFrame([new_row]).to_csv(csv_path,
                                       mode='a',
                                       header=False,
                                       index=False)

        print(f"Processed id {index}: {context_list}")

    return


def main(method="generation"):
    llm = LLMClient()
    df = pd.read_csv('dataset/chats_data.csv')
    if method == "generation":
        generation(llm, df)
    else:
        transformation(llm, df)


if __name__ == "__main__":
    main("transformation")
