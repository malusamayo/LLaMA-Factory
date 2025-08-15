import json
import pandas as pd

with open("./data/oss_fakeclue_image_val.json", "r") as f:
    val_json = json.loads(f.read())
df_fakeclue = pd.DataFrame(val_json)

df_fakeclue['image'] = df_fakeclue['images'].map(lambda x: x[0].replace('ff/', 'ff++/').replace('fakeclue/test/', ''))

with open("./data/oss_fakeclue_image_val_annotated.json", "r") as f:
    df_fakeclue_annotated = pd.DataFrame(json.loads(f.read()))

df_fakeclue = df_fakeclue.merge(df_fakeclue_annotated[['image', 'cate']], on='image', how='left')
df_fakeclue['category'] = df_fakeclue['cate']

df_loki = pd.read_csv("./data/oss_loki_image.csv")


def get_preds_for_explain(x):
    item = x.split(".")[0]
    if 'fake' in item:
        return 'yes'
    elif 'real' in item:
        return 'no'
    else:
        return None


def compute_metrics(df):
    """
    Compute TP, FP, TN, FN, Recall, Precision, and F1 score for binary classification.

    Args:
        df (pd.DataFrame): DataFrame with columns 'model_output' and 'label', both containing 'yes' or 'no'.

    Returns:
        dict: Dictionary with keys 'TP', 'FP', 'TN', 'FN', 'R', 'P', 'F1'
    """
    TP = (
        (df["model_output"] == "yes") & (df["descriptive_two_class_label"] == "yes")
    ).sum()
    FP = (
        (df["model_output"] == "yes") & (df["descriptive_two_class_label"] == "no")
    ).sum()
    TN = (
        (df["model_output"] == "no") & (df["descriptive_two_class_label"] == "no")
    ).sum()
    FN = (
        (df["model_output"] == "no") & (df["descriptive_two_class_label"] == "yes")
    ).sum()

    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) 

    return pd.Series(
        {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Precision": P,
            "Recall": R,
            "F1": F1,
            "Accuracy": acc,
        }
    )


import json
import pandas as pd

results = {
    # 'qwen2_5vl-3b': "/home/ychenyang/LLaMA-Factory/generated_predictions_qwen2_vl.jsonl",
    # # 'qwen2_5vl-3b': "/home/ychenyang/LLaMA-Factory/generated_predictions_val_qwen2_vl.jsonl",
    # 'gemma-3-4b': "/home/ychenyang/LLaMA-Factory/generated_predictions_gemma-3-4b.jsonl",

    # 'qwen2_5vl-3b': "/home/ychenyang/LLaMA-Factory/generated_predictions_fakeclue_qwen2_5vl-3b.jsonl",
    # 'gemma-3-4b': "/home/ychenyang/LLaMA-Factory/generated_predictions_fakeclue_gemma-3-4b.jsonl",

    # "qwen2_5vl-3": "/home/ychenyang/LLaMA-Factory/generated_predictions_loki_qwen2_5vl-3b.jsonl",
    # 'gemma-3-4b': "/home/ychenyang/LLaMA-Factory/generated_predictions_loki_gemma-3-4b.jsonl",

    # "exp_qwen2_5vl-3": "/home/ychenyang/LLaMA-Factory/generated_predictions_fakeclue_exp_eval_qwen2_5vl-3b.jsonl",
    # 'exp_gemma-3-4b': "/home/ychenyang/LLaMA-Factory/generated_predictions_fakeclue_exp_eval_gemma-3-4b.jsonl",
    # "exp_llama3-11b": "/home/ychenyang/LLaMA-Factory/generated_predictions_fakeclue_eval_exp_llama3_11b_2epoch.jsonl",

    # "exp_qwen2_5vl-3": "/home/ychenyang/LLaMA-Factory/generated_predictions_loki_exp_qwen2_5vl-3b.jsonl",
    # 'exp_gemma-3-4b': "/home/ychenyang/LLaMA-Factory/generated_predictions_loki_exp_gemma-3-4b.jsonl",
    # "exp_llama3-11b": "/home/ychenyang/LLaMA-Factory/generated_predictions_loki_exp_llama3_11b_2epoch.jsonl",

    'gemma-3-4b': "/home/ychenyang/LLaMA-Factory/results/fakeclue/generated_predictions_fakeclue_eval_gemma-3-4b_2epoch.jsonl",
    "qwen2_5vl-3": "/home/ychenyang/LLaMA-Factory/results/fakeclue/generated_predictions_fakeclue_eval_qwen2_5vl-3b_2epoch.jsonl",
    "llama3-11b": "/home/ychenyang/LLaMA-Factory/results/fakeclue/generated_predictions_fakeclue_eval_llama3_11b_2epoch.jsonl",

    # 'gemma-3-4b': "/home/ychenyang/LLaMA-Factory/results/loki/generated_predictions_loki_gemma-3-4b_2epoch.jsonl",
    # "qwen2_5vl-3": "/home/ychenyang/LLaMA-Factory/results/loki/generated_predictions_loki_qwen2_5vl-3b_2epoch.jsonl",
    # "llama3-11b": "/home/ychenyang/LLaMA-Factory/results/loki/generated_predictions_loki_llama3_11b_2epoch.jsonl",
}


aigc_dfs = {}

for k, path in results.items():
    with open(path, "r") as f:
        lines = f.read().splitlines()

    lines = [json.loads(line) for line in lines]
    df = pd.DataFrame(lines)
    if 'exp' in k:
        df['model_output'] = df['predict'].map(get_preds_for_explain)
        # df['label'] = df['label'].map(get_preds_for_explain).str.strip()
    else:
        df['model_output'] = df['predict']
    df['label'] = df['label'].str.strip()
    df['descriptive_two_class_label'] = df['label']
    if 'loki' in path:
        aigc_dfs[k] = pd.concat([df, df_loki['category']], axis=1)
    elif 'fakeclue' in path:
        aigc_dfs[k] = pd.concat([df, df_fakeclue], axis=1)

for k, df in aigc_dfs.items():
    print(k)
    for group_key in ['category']:
        grouped_metrics = df.groupby(group_key).apply(compute_metrics).reset_index()
        grouped_metrics.loc[len(grouped_metrics)] = compute_metrics(df).to_dict() | {
            group_key: "all"
        }
        print(grouped_metrics)