import os
import json
import glob
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForMultipleChoice


DISTANCES = [20, 50, 100, 200]
DENSITIES = [1, 5, 10, 20]
SEEDS = [1, 2, 3]


def render_dialogue(sample):
    text = ""
    spans = []

    for turn in sample["dialogue"]:
        role = turn.get("role", "unknown")
        turn_text = f'{turn["speaker"]}: {turn["text"]}\n'

        start = len(text)
        text += turn_text
        end = len(text)

        spans.append({"role": role, "start": start, "end": end})

    return text, spans


def get_token_indices_by_role(offsets, spans, role):
    token_ids = []

    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue

        for span in spans:
            if span["role"] == role and s < span["end"] and e > span["start"]:
                token_ids.append(i)
                break

    return token_ids


def entropy(attn):
    eps = 1e-12
    attn = attn / (attn.sum() + eps)
    return float(-(attn * torch.log(attn + eps)).sum().item())


def load_model_and_tokenizer(model_name, checkpoint_path, device):
    if model_name == "bert":
        hf_name = "bert-base-uncased"
    elif model_name == "longformer":
        hf_name = "allenai/longformer-base-4096"
    else:
        hf_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(hf_name)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    return model, tokenizer


def analyze_one_file(
    model,
    tokenizer,
    model_name,
    test_path,
    output_dir,
    device,
    max_length,
    limit=None,
    save_heatmap=False,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(test_path, "r") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    rows = []
    heatmap_saved = 0

    for sample in data:
        dialogue_text, spans = render_dialogue(sample)

        choices = [
            dialogue_text + "\nCandidate: " + c
            for c in sample["candidates"]
        ]

        enc = tokenizer(
            choices,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = enc.pop("offset_mapping")

        input_ids = enc["input_ids"].unsqueeze(0).to(device)
        attention_mask = enc["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        pred = int(out.logits.argmax(dim=-1).item())
        correct = int(pred == sample["correct_index"])

        chosen_offsets = offsets[pred].tolist()

        signal_ids = get_token_indices_by_role(chosen_offsets, spans, "signal")
        distractor_ids = get_token_indices_by_role(chosen_offsets, spans, "distractor")
        noise_ids = get_token_indices_by_role(chosen_offsets, spans, "noise")

        layer_signal_mass = []
        layer_entropy = []

        for layer_attn in out.attentions:
            if layer_attn.dim() == 4:
                attn = layer_attn[pred]
            elif layer_attn.dim() == 5:
                attn = layer_attn[0, pred]
            else:
                raise ValueError(f"Unexpected attention shape: {layer_attn.shape}")

            cls_attn = attn[:, 0, :].mean(dim=0).detach().cpu()

            sig_mass = cls_attn[signal_ids].sum().item() if signal_ids else 0.0
            layer_signal_mass.append(sig_mass)
            layer_entropy.append(entropy(cls_attn))

        last_layer = out.attentions[-1]

        if last_layer.dim() == 4:
            attn = last_layer[pred]
        else:
            attn = last_layer[0, pred]

        cls_attn = attn[:, 0, :].mean(dim=0).detach().cpu()

        signal_mass = cls_attn[signal_ids].sum().item() if signal_ids else 0.0
        distractor_mass = cls_attn[distractor_ids].sum().item() if distractor_ids else 0.0
        noise_mass = cls_attn[noise_ids].sum().item() if noise_ids else 0.0
        attn_entropy = entropy(cls_attn)

        row = {
            "sample_id": sample.get("sample_id"),
            "model": model_name,
            "target_distance_tokens": sample.get("target_distance_tokens"),
            "actual_distance_tokens": sample.get("actual_distance_tokens"),
            "density": sample.get("density"),
            "correct_index": sample["correct_index"],
            "pred_index": pred,
            "is_correct": correct,
            "signal_attention_mass": signal_mass,
            "distractor_attention_mass": distractor_mass,
            "noise_attention_mass": noise_mass,
            "attention_entropy": attn_entropy,
        }

        for i, value in enumerate(layer_signal_mass):
            row[f"layer_{i}_signal_mass"] = value

        for i, value in enumerate(layer_entropy):
            row[f"layer_{i}_entropy"] = value

        rows.append(row)

        if save_heatmap and heatmap_saved < 3:
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0, pred].detach().cpu())
            save_attention_heatmap(
                tokens=tokens,
                attn_vec=cls_attn.numpy(),
                title=f"{model_name} | {sample.get('sample_id')} | correct={correct}",
                save_path=os.path.join(
                    output_dir,
                    f"heatmap_{heatmap_saved}_{sample.get('sample_id')}.png",
                ),
            )
            heatmap_saved += 1

        del out
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "attention_metrics.csv")
    df.to_csv(csv_path, index=False)

    return df


def save_attention_heatmap(tokens, attn_vec, title, save_path, max_tokens=120):
    tokens = tokens[:max_tokens]
    attn_vec = attn_vec[:max_tokens]

    plt.figure(figsize=(16, 3))
    plt.imshow([attn_vec], aspect="auto")
    plt.yticks([0], ["CLS attention"])
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_results(all_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    all_df.to_csv(os.path.join(save_dir, "all_attention_metrics.csv"), index=False)

    summary = all_df.groupby(
        ["model", "target_distance_tokens", "density"]
    ).agg(
        accuracy=("is_correct", "mean"),
        signal_attention_mass=("signal_attention_mass", "mean"),
        distractor_attention_mass=("distractor_attention_mass", "mean"),
        noise_attention_mass=("noise_attention_mass", "mean"),
        attention_entropy=("attention_entropy", "mean"),
    ).reset_index()

    summary.to_csv(os.path.join(save_dir, "attention_summary.csv"), index=False)

    plt.figure(figsize=(7, 5))
    for model_name in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model_name]
        line = sub.groupby("target_distance_tokens")["signal_attention_mass"].mean().reset_index()
        plt.plot(line["target_distance_tokens"], line["signal_attention_mass"], marker="o", label=model_name)

    plt.xlabel("Signal-query distance")
    plt.ylabel("Signal attention mass")
    plt.title("Signal Attention Mass vs. Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "signal_attention_mass_vs_distance.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for model_name in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model_name]
        line = sub.groupby("density")["attention_entropy"].mean().reset_index()
        line = line.sort_values("density")
        plt.plot(line["density"], line["attention_entropy"], marker="o", label=model_name)

    plt.xlabel("Density")
    plt.ylabel("Attention entropy")
    plt.title("Attention Entropy vs. Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attention_entropy_vs_density.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for model_name in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model_name]
        plt.scatter(sub["signal_attention_mass"], sub["accuracy"], label=model_name)

    plt.xlabel("Signal attention mass")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Signal Attention Mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_signal_attention_mass.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for model_name in sorted(summary["model"].unique()):
        sub = summary[summary["model"] == model_name]
        plt.scatter(sub["attention_entropy"], sub["accuracy"], label=model_name)

    plt.xlabel("Attention entropy")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Attention Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_vs_attention_entropy.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--grid_base", required=True)
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.checkpoint_path,
        device,
    )

    all_rows = []

    for d in DISTANCES:
        for den in DENSITIES:
            for seed in SEEDS:
                test_path = os.path.join(
                    args.grid_base,
                    f"d{d}_den{den}_seed{seed}",
                    "test.json",
                )

                if not os.path.exists(test_path):
                    print(f"Missing test file: {test_path}")
                    continue

                output_dir = os.path.join(
                    args.output_base,
                    args.model,
                    f"d{d}_den{den}_seed{seed}",
                    "attention_analysis",
                )

                save_heatmap = (
                    (d == 20 and den == 1 and seed == 1)
                    or (d == 200 and den == 20 and seed == 1)
                )

                print(f"Analyzing {args.model} | d={d} | den={den} | seed={seed}")

                df = analyze_one_file(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=args.model,
                    test_path=test_path,
                    output_dir=output_dir,
                    device=device,
                    max_length=args.max_length,
                    limit=args.limit,
                    save_heatmap=save_heatmap,
                )

                all_rows.append(df)

    if len(all_rows) == 0:
        print("No attention results found.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)

    fig_dir = os.path.join(args.output_base, "attention_figures")
    plot_results(all_df, fig_dir)

    print("Done.")
    print(f"Saved final figures and CSV files to: {fig_dir}")


if __name__ == "__main__":
    main()