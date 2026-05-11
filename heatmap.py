import os
import json
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_BASE = "experiment_outputs"
MODELS = ["bert", "longformer"]
DISTANCES = [20, 50, 100, 200]
DENSITIES = [1, 5, 10, 20]
SEEDS = [1, 2, 3]


def find_accuracy(result_dir):
    possible_files = [
        "metrics.json",
        "result.json",
        "results.json",
        "test_results.json",
        "eval_results.json"
    ]

    for file in possible_files:
        path = os.path.join(result_dir, file)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)

            for key in ["accuracy", "acc", "test_accuracy"]:
                if key in data:
                    return data[key]

    raise FileNotFoundError(f"No accuracy file found in {result_dir}")


all_rows = []

for model in MODELS:
    rows = []

    for d in DISTANCES:
        for den in DENSITIES:
            accs = []

            for s in SEEDS:
                result_dir = f"{OUTPUT_BASE}/{model}/d{d}_den{den}_seed{s}"

                try:
                    acc = find_accuracy(result_dir)
                    accs.append(acc)

                    rows.append({
                        "model": model,
                        "distance": d,
                        "density": den,
                        "seed": s,
                        "accuracy": acc
                    })

                except Exception as e:
                    print(f"Missing result: model={model}, d={d}, den={den}, seed={s}")
                    print(e)

            if accs:
                all_rows.append({
                    "model": model,
                    "distance": d,
                    "density": den,
                    "mean_accuracy": sum(accs) / len(accs),
                    "std_accuracy": pd.Series(accs).std()
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_BASE}/{model}_grid_results.csv", index=False)

    mean_df = (
        df.groupby(["distance", "density"])["accuracy"]
        .mean()
        .reset_index()
    )

    heatmap_data = mean_df.pivot(
        index="distance",
        columns="density",
        values="accuracy"
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(heatmap_data, aspect="auto")
    plt.colorbar(label="Mean Accuracy")

    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.xlabel("Distractor Density")
    plt.ylabel("Signal-Query Distance")
    plt.title(f"{model.upper()} Accuracy Heatmap")

    for i, d in enumerate(heatmap_data.index):
        for j, den in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[d, den]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_BASE}/{model}_heatmap.png", dpi=300)
    plt.close()


summary_df = pd.DataFrame(all_rows)
summary_df.to_csv(f"{OUTPUT_BASE}/comparison_results.csv", index=False)

compare = summary_df.pivot_table(
    index=["distance", "density"],
    columns="model",
    values="mean_accuracy"
).reset_index()

compare["bert_minus_longformer"] = compare["bert"] - compare["longformer"]
compare.to_csv(f"{OUTPUT_BASE}/comparison_results.csv", index=False)

heatmap_compare = compare.pivot(
    index="distance",
    columns="density",
    values="bert_minus_longformer"
)

plt.figure(figsize=(7, 5))
plt.imshow(heatmap_compare, aspect="auto")
plt.colorbar(label="BERT - Longformer Accuracy")

plt.xticks(range(len(heatmap_compare.columns)), heatmap_compare.columns)
plt.yticks(range(len(heatmap_compare.index)), heatmap_compare.index)

plt.xlabel("Distractor Density")
plt.ylabel("Signal-Query Distance")
plt.title("BERT vs Longformer Accuracy Difference")

for i, d in enumerate(heatmap_compare.index):
    for j, den in enumerate(heatmap_compare.columns):
        value = heatmap_compare.loc[d, den]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_BASE}/comparison_heatmap.png", dpi=300)
plt.close()