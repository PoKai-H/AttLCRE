import os
import json
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_BASE = "data_experiment/experiment_output"

MODELS = ["bert", "longformer"]
DISTANCES = [20, 50, 100, 200]
DENSITIES = [1, 5, 10, 20]
SEEDS = [1, 2, 3]


def read_accuracy(result_dir):
    pred_path = os.path.join(result_dir, "test_predictions.json")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"No test_predictions.json found in {result_dir}")

    with open(pred_path, "r") as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError(f"Empty test_predictions.json in {result_dir}")

    correct = sum(item["correct"] for item in data)
    total = len(data)

    return correct / total


for model in MODELS:
    rows = []

    for d in DISTANCES:
        for den in DENSITIES:
            for seed in SEEDS:
                result_dir = f"{OUTPUT_BASE}/{model}/d{d}_den{den}_seed{seed}"

                try:
                    acc = read_accuracy(result_dir)

                    rows.append({
                        "model": model,
                        "distance": d,
                        "density": den,
                        "seed": seed,
                        "accuracy": acc,
                    })

                    print(f"Loaded: {model}, d={d}, den={den}, seed={seed}, acc={acc:.4f}")

                except Exception as e:
                    print(f"Missing result: {model}, d={d}, den={den}, seed={seed}")
                    print(e)

    df = pd.DataFrame(rows)

    model_output_dir = f"{OUTPUT_BASE}/{model}"
    os.makedirs(model_output_dir, exist_ok=True)

    csv_path = f"{model_output_dir}/{model}_grid_results.csv"
    df.to_csv(csv_path, index=False)

    if df.empty:
        print(f"No valid results found for {model}. Skip summary and heatmap.")
        continue

    summary = (
        df.groupby(["distance", "density"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_path = f"{model_output_dir}/{model}_grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    heatmap_data = summary.pivot(
        index="distance",
        columns="density",
        values="mean"
    )

    plt.figure(figsize=(7, 5))
    plt.imshow(heatmap_data, aspect="auto")
    plt.colorbar(label="Mean Accuracy")

    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

    plt.xlabel("Distractor Density")
    plt.ylabel("Signal-Query Distance")
    plt.title(f"{model.upper()} 4x4 Grid Accuracy Heatmap")

    for i, distance in enumerate(heatmap_data.index):
        for j, density in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[distance, density]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center")

    plt.tight_layout()

    heatmap_path = f"{model_output_dir}/{model}_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Saved results for {model}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Heatmap: {heatmap_path}")