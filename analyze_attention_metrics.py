import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    LongformerForSequenceClassification,
)


def load_data(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def get_model_and_tokenizer(model_name, checkpoint_path, device):
    if model_name == "bert":
        base = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = BertForSequenceClassification.from_pretrained(
            base,
            num_labels=2,
            output_attentions=True,
            attn_implementation="eager"
        )

    elif model_name == "longformer":
        base = "allenai/longformer-base-4096"
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = LongformerForSequenceClassification.from_pretrained(
            base,
            num_labels=2,
            output_attentions=True,
            attn_implementation="eager"
        )

    else:
        raise ValueError("model_name must be bert or longformer")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    return model, tokenizer


def render_text(example, include_speaker=True):
    parts = []

    for turn in example.get("dialogue", []):
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")

        if include_speaker:
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(text)

    query = example.get("query", "")
    if query:
        parts.append(f"Query: {query}")

    return "\n".join(parts)


def get_candidates(example):
    if "candidates" in example:
        return example["candidates"]
    if "options" in example:
        return example["options"]
    raise KeyError("Cannot find candidates/options in example.")


def get_label(example):
    for key in ["label", "answer", "correct_idx", "correct_index"]:
        if key in example:
            return int(example[key])
    raise KeyError("Cannot find label in example.")


def find_signal_char_spans(example, full_text):
    spans = []

    for turn in example.get("dialogue", []):
        role = turn.get("role", "")
        text = turn.get("text", "")

        if role == "signal" and text:
            start = full_text.find(text)
            if start != -1:
                end = start + len(text)
                spans.append((start, end))

    return spans


def char_spans_to_token_mask(offset_mapping, signal_spans):
    mask = []

    for start, end in offset_mapping:
        is_signal = False

        for s_start, s_end in signal_spans:
            if start < s_end and end > s_start:
                is_signal = True
                break

        mask.append(is_signal)

    return torch.tensor(mask, dtype=torch.bool)


def attention_entropy(attn_vec):
    eps = 1e-12
    attn_vec = attn_vec + eps
    return float(-(attn_vec * torch.log(attn_vec)).sum().item())


def analyze_one_example(
    example,
    model,
    tokenizer,
    model_name,
    device,
    max_length,
    include_speaker=True,
):
    context = render_text(example, include_speaker=include_speaker)
    candidates = get_candidates(example)
    label = get_label(example)

    signal_spans = find_signal_char_spans(example, context)

    encoded = tokenizer(
        [context] * len(candidates),
        [str(c) for c in candidates],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop("offset_mapping")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if model_name == "longformer":
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        model_inputs["global_attention_mask"] = global_attention_mask

    with torch.no_grad():
        outputs = model(**model_inputs)

    logits = outputs.logits

    positive_scores = logits[:, 1]
    pred = int(torch.argmax(positive_scores).item())
    correct = int(pred == label)

    attentions = outputs.attentions

    correct_choice = label
    choice_offsets = offset_mapping[correct_choice]

    sequence_ids = encoded.sequence_ids(correct_choice)

    signal_mask_list = []

    for i, (start, end) in enumerate(choice_offsets.tolist()):
        if sequence_ids[i] != 0:
            signal_mask_list.append(False)
            continue

        is_signal = False
        for s_start, s_end in signal_spans:
            if start < s_end and end > s_start:
                is_signal = True
                break

        signal_mask_list.append(is_signal)

    signal_mask = torch.tensor(signal_mask_list, dtype=torch.bool).to(device)
    valid_mask = attention_mask[correct_choice].bool()

    # ===== FIX LONGFORMER LENGTH MISMATCH =====
    min_len = min(len(signal_mask), len(valid_mask))

    signal_mask = signal_mask[:min_len]
    valid_mask = valid_mask[:min_len]
    # ==========================================


    signal_mask = signal_mask & valid_mask

    layer_results = []

    for layer_idx, layer_attn in enumerate(attentions):

        attn = layer_attn[correct_choice]

        attn_mean = attn.mean(dim=0)

        cls_attn = attn_mean[0]

        cls_attn = cls_attn[:min_len]

        cls_attn = cls_attn * valid_mask
        cls_attn = cls_attn / (cls_attn.sum() + 1e-12)

        if signal_mask.sum().item() > 0:
            signal_mass = float(cls_attn[signal_mask].sum().item())
        else:
            signal_mass = np.nan

        entropy = attention_entropy(cls_attn[valid_mask])

        layer_results.append({
            "layer": layer_idx,
            "signal_attention_mass": signal_mass,
            "attention_entropy": entropy,
        })

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[correct_choice].detach().cpu().tolist()
    )

    attention_matrix = attentions[-1][correct_choice].mean(dim=0).detach().cpu().numpy()

    return {
        "correct": correct,
        "pred": pred,
        "label": label,
        "layer_results": layer_results,
        "context": context,
        "signal_spans": signal_spans,
        "tokens": tokens,
        "attention_matrix": attention_matrix,
    }


def summarize_condition(rows):
    df = pd.DataFrame(rows)

    summary = {
        "accuracy": df["correct"].mean(),
        "num_samples": len(df),
    }

    layer_df = df.explode("layer_results").reset_index(drop=True)
    expanded = pd.json_normalize(layer_df["layer_results"])

    layer_df = pd.concat(
        [layer_df.drop(columns=["layer_results"]), expanded],
        axis=1
    )

    final_layer = layer_df["layer"].max()
    final_df = layer_df[layer_df["layer"] == final_layer]

    summary["signal_attention_mass_mean"] = final_df["signal_attention_mass"].mean()
    summary["signal_attention_mass_std"] = final_df["signal_attention_mass"].std()
    summary["attention_entropy_mean"] = final_df["attention_entropy"].mean()
    summary["attention_entropy_std"] = final_df["attention_entropy"].std()

    return summary, layer_df


def save_attention_heatmap(case_result, save_path, title, max_tokens=80):
    tokens = case_result["tokens"]
    attn = case_result["attention_matrix"]

    tokens = tokens[:max_tokens]
    attn = attn[:max_tokens, :max_tokens]

    plt.figure(figsize=(12, 10))
    plt.imshow(attn, aspect="auto")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    plt.yticks(range(len(tokens)), tokens, fontsize=6)
    plt.title(title)
    plt.colorbar(label="attention")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_signal_mass(metrics_df, save_path):
    plt.figure(figsize=(8, 5))

    for density in sorted(metrics_df["density"].unique()):
        sub = metrics_df[metrics_df["density"] == density]
        sub = sub.sort_values("distance")
        plt.plot(
            sub["distance"],
            sub["signal_attention_mass_mean"],
            marker="o",
            label=f"density={density}"
        )

    plt.xlabel("Signal-query distance")
    plt.ylabel("Signal attention mass")
    plt.title("Signal Attention Mass vs. Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_entropy(metrics_df, save_path):
    plt.figure(figsize=(8, 5))

    for distance in sorted(metrics_df["distance"].unique()):
        sub = metrics_df[metrics_df["distance"] == distance]
        sub = sub.sort_values("density")
        plt.plot(
            sub["density"],
            sub["attention_entropy_mean"],
            marker="o",
            label=f"distance={distance}"
        )

    plt.xlabel("Distractor density")
    plt.ylabel("Attention entropy")
    plt.title("Attention Entropy vs. Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["bert", "longformer"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--grid_base", type=str, required=True)
    parser.add_argument("--output_base", type=str, required=True)

    parser.add_argument("--distances", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--densities", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include_speaker", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_base, exist_ok=True)

    model, tokenizer = get_model_and_tokenizer(
        args.model,
        args.checkpoint_path,
        device
    )

    all_condition_summaries = []
    all_layer_rows = []

    easy_case = None
    hard_case = None

    for distance in args.distances:
        for density in args.densities:
            for seed in args.seeds:
                file_path = os.path.join(
                    args.grid_base,
                    f"d{distance}_den{density}_seed{seed}",
                    f"{args.split}.json"
                )

                if not os.path.exists(file_path):
                    print(f"[SKIP] File not found: {file_path}")
                    continue

                print(f"\nAnalyzing {args.model} | distance={distance} | density={density} | seed={seed}")

                data = load_data(file_path, args.limit)
                rows = []

                for idx, example in enumerate(tqdm(data)):
                    try:
                        result = analyze_one_example(
                            example=example,
                            model=model,
                            tokenizer=tokenizer,
                            model_name=args.model,
                            device=device,
                            max_length=args.max_length,
                            include_speaker=args.include_speaker,
                        )

                        rows.append({
                            "model": args.model,
                            "distance": distance,
                            "density": density,
                            "seed": seed,
                            "sample_id": idx,
                            "correct": result["correct"],
                            "pred": result["pred"],
                            "label": result["label"],
                            "layer_results": result["layer_results"],
                        })

                        if distance == min(args.distances) and density == min(args.densities) and easy_case is None:
                            easy_case = result

                        if distance == max(args.distances) and density == max(args.densities) and hard_case is None:
                            hard_case = result

                    except Exception as e:
                        print(f"[ERROR] sample {idx}: {e}")

                if len(rows) == 0:
                    continue

                summary, layer_df = summarize_condition(rows)

                summary.update({
                    "model": args.model,
                    "distance": distance,
                    "density": density,
                    "seed": seed,
                })

                all_condition_summaries.append(summary)

                layer_df["model"] = args.model
                layer_df["distance"] = distance
                layer_df["density"] = density
                layer_df["seed"] = seed

                all_layer_rows.append(layer_df)

    metrics_df = pd.DataFrame(all_condition_summaries)

    metrics_path = os.path.join(
        args.output_base,
        f"{args.model}_attention_grid_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)

    if all_layer_rows:
        layer_all_df = pd.concat(all_layer_rows, ignore_index=True)
        layer_path = os.path.join(
            args.output_base,
            f"{args.model}_layer_attention_metrics.csv"
        )
        layer_all_df.to_csv(layer_path, index=False)

    mean_df = metrics_df.groupby(
        ["model", "distance", "density"],
        as_index=False
    ).agg({
        "accuracy": ["mean", "std"],
        "signal_attention_mass_mean": "mean",
        "attention_entropy_mean": "mean",
        "num_samples": "sum",
    })

    mean_df.columns = [
        "model",
        "distance",
        "density",
        "accuracy_mean",
        "accuracy_std",
        "signal_attention_mass_mean",
        "attention_entropy_mean",
        "num_samples",
    ]

    mean_path = os.path.join(
        args.output_base,
        f"{args.model}_attention_grid_summary.csv"
    )
    mean_df.to_csv(mean_path, index=False)

    plot_signal_mass(
        mean_df,
        os.path.join(args.output_base, f"{args.model}_signal_mass_vs_distance.png")
    )

    plot_entropy(
        mean_df,
        os.path.join(args.output_base, f"{args.model}_entropy_vs_density.png")
    )

    if easy_case is not None:
        save_attention_heatmap(
            easy_case,
            os.path.join(args.output_base, f"{args.model}_easy_case_attention_heatmap.png"),
            f"{args.model} Easy Case Attention Heatmap"
        )

    if hard_case is not None:
        save_attention_heatmap(
            hard_case,
            os.path.join(args.output_base, f"{args.model}_hard_case_attention_heatmap.png"),
            f"{args.model} Hard Case Attention Heatmap"
        )

    print("\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary to: {mean_path}")


if __name__ == "__main__":
    main()