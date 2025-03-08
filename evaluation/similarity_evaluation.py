import pandas as pd
import textdistance


def calculate_similarity(text1, text2):
    """
    计算两个文本之间的四个相似度指标
    返回: (levenshtein, jaccard, cosine, damerau_levenshtein)
    """
    # 处理缺失值并确保字符串类型
    text1 = str(text1) if not pd.isna(text1) else ""
    text2 = str(text2) if not pd.isna(text2) else ""

    return (
        textdistance.levenshtein.normalized_similarity(text1, text2),
        textdistance.jaccard.normalized_similarity(text1.split(), text2.split()),
        textdistance.cosine.normalized_similarity(text1.split(), text2.split()),
        textdistance.damerau_levenshtein.normalized_similarity(text1, text2)
    )


def main(method):
    df = pd.read_csv(f"../results/{method}_data.csv")

    # 处理缺失值填充（可选）
    df["system"] = df["system"].fillna("").astype(str)
    df["new_system"] = df["new_system"].fillna("").astype(str)
    df["user"] = df["user"].fillna("").astype(str)
    df["new_user"] = df["new_user"].fillna("").astype(str)

    # 计算相似度
    df[["lev_sys", "jac_sys", "cos_sys", "dam_sys"]] = df.apply(
        lambda row: calculate_similarity(row["system"], row["new_system"]),
        axis=1,
        result_type="expand"
    )

    df[["lev_user", "jac_user", "cos_user", "dam_user"]] = df.apply(
        lambda row: calculate_similarity(row["user"], row["new_user"]),
        axis=1,
        result_type="expand"
    )

    # 修正drop和mean的语法
    df = df.drop(columns=['intent', 'id', 'user_bf', 'system_bf'])
    print(df[["lev_sys", "jac_sys", "cos_sys", "dam_sys", "lev_user", "jac_user", "cos_user", "dam_user"]].mean())
    df.to_csv(f"../results/{method}_similarity.csv", index=False)


if __name__ == "__main__":
    main("transformation")
    """GENERATION
    lev_sys     0.204222
    jac_sys     0.046651
    cos_sys     0.087856
    dam_sys     0.204530
    lev_user    0.254915
    jac_user    0.096785
    cos_user    0.168399
    dam_user    0.255122
    dtype: float64
    """
    """TRANSFORMATION
    lev_sys     0.862492
    jac_sys     0.827601
    cos_sys     0.852756
    dam_sys     0.862555
    lev_user    0.852610
    jac_user    0.792397
    cos_user    0.837228
    dam_user    0.852648
    dtype: float64
    """