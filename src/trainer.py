from __future__ import annotations

from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from src.eval import (
    accuracy_from_predictions,
    collect_candidate_scores,
    compute_sample_level_predictions,
)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    log_every = max(1, len(dataloader) // 10)

    print(f"\nStarting epoch {epoch}/{num_epochs} ({len(dataloader)} train batches)")

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_bias = batch.get("attention_bias")
        if attention_bias is not None:
            attention_bias = attention_bias.to(device)

        optimizer.zero_grad()

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        if attention_bias is not None:
            model_inputs["attention_bias"] = attention_bias

        outputs = model(**model_inputs)

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # restricting too large gradient for training stability

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(dataloader):
            avg_loss = total_loss / batch_idx
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e}",
                flush=True,
            )

    return total_loss / max(len(dataloader), 1)


def run_training(
    model,
    train_loader,
    val_loader,
    device: str,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    print(
        "\nTraining setup | "
        f"Epochs: {num_epochs} | "
        f"Train batches/epoch: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Total steps: {total_steps} | "
        f"Warmup steps: {warmup_steps} | "
        f"Device: {device}",
        flush=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    ) # first warmup and than linearly degrade learning rate

    best_val_acc = -1.0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
        )

        print(f"Evaluating epoch {epoch + 1}/{num_epochs}...", flush=True)
        val_candidate_outputs = collect_candidate_scores(model, val_loader, device)
        val_predictions = compute_sample_level_predictions(val_candidate_outputs)
        val_acc = accuracy_from_predictions(val_predictions)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Sample Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}", flush=True)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
