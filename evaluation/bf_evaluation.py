import ast
import warnings
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer


warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")
model = AutoModelForSequenceClassification.from_pretrained("KevSun/Personality_LM", ignore_mismatched_sizes=True)


def personality_detection(text):
    encoded_input = tokenizer(str(text),
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=64)
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_scores = predictions[0].tolist()

    # 新模型原始输出顺序
    original_order = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]

    # 调整为原代码要求的顺序和命名约定
    order_mapping = {
        'Openness': predicted_scores[1],
        'Conscientiousness': predicted_scores[2],
        'Extroversion': predicted_scores[3],
        'Agreeableness': predicted_scores[0],
        'Neuroticism': predicted_scores[4]
    }

    # 保持与原代码一致的输出顺序
    new_order = ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']
    return [order_mapping[trait] for trait in new_order]


def evl(method):
    if method == "generation":
        input_path = "../results/generation_data.csv"
        output_path = "../results/generation_bf_evaluation.csv"
    else:
        input_path = "../results/transformation_data.csv"
        output_path = "../results/transformation_bf_evaluation.csv"

    df = pd.read_csv(input_path)
    df["system_bf"] = df["system_bf"].apply(ast.literal_eval)
    df["user_bf"] = df["user_bf"].apply(ast.literal_eval)

    df["system_bert_bf"] = df["new_system"].apply(personality_detection)
    df["user_bert_bf"] = df["new_user"].apply(personality_detection)

    # 残差计算保持不变
    df["system_residual"] = df.apply(
        lambda row: [s - b for s, b in zip(row["system_bf"], row["system_bert_bf"])],
        axis=1
    )
    df["user_residual"] = df.apply(
        lambda row: [u - b for u, b in zip(row["user_bf"], row["user_bert_bf"])],
        axis=1
    )
    def calculate_stats(residual_column, title):
        # 初始化字典收集各维度误差
        trait_names = ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']
        residuals = {trait: [] for trait in trait_names}
        total_errors = []

        # 遍历所有残差记录
        for res_list in residual_column:
            for i, trait in enumerate(trait_names):
                residuals[trait].append(abs(res_list[i]))  # 使用绝对值计算误差
                total_errors.append(abs(res_list[i]))

        # 计算平均误差
        avg_by_trait = {t: sum(vals) / len(vals) for t, vals in residuals.items()}
        overall_avg = sum(total_errors) / len(total_errors) if total_errors else 0

        # 打印结果
        print(f"\n{title} 平均误差:")
        for trait in trait_names:
            print(f"{trait}: {avg_by_trait[trait]:.4f}")
        print(f"总平均误差: {overall_avg:.4f}\n")

    # 执行统计
    calculate_stats(df["system_residual"], "机器性格预测")
    calculate_stats(df["user_residual"], "用户性格预测")

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    evl("generation")
    evl("transformation")
    """GENERATION
    机器性格预测 平均误差:
    Openness: 0.5452
    Conscientiousness: 0.4396
    Extroversion: 0.3685
    Agreeableness: 0.5397
    Neuroticism: 0.1654
    总平均误差: 0.4117
    
    
    用户性格预测 平均误差:
    Openness: 0.5511
    Conscientiousness: 0.3728
    Extroversion: 0.4348
    Agreeableness: 0.5930
    Neuroticism: 0.1732
    总平均误差: 0.4250
    """
    """ TRANSFORMATION
    机器性格预测 平均误差:
    Openness: 0.4622
    Conscientiousness: 0.4783
    Extroversion: 0.1835
    Agreeableness: 0.3907
    Neuroticism: 0.1060
    总平均误差: 0.3241
    
    
    用户性格预测 平均误差:
    Openness: 0.4346
    Conscientiousness: 0.2209
    Extroversion: 0.3081
    Agreeableness: 0.4831
    Neuroticism: 0.1595
    总平均误差: 0.3212
    """