"""
train.py
--------
Fine-tuning DeBERTa-v3-base on PAN 2012 grooming detection.

Metric: F_beta (beta=2) -- recall weighted 2x over precision.
Threshold: calibrated on val set each epoch, not fixed at 0.5.

Usage:
  python train.py
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import json

BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BETA = 2.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    denom = beta * beta * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denom


def compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(torch.tensor(logits[:, 1])).numpy()
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall":    recall,
        "f2":        f_beta(precision, recall),
        "tp": tp, "fp": fp, "fn": fn,
        "threshold": threshold,
    }


def calibrate_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    """Grid search [0.05, 0.95] step 0.01 maximising F2."""
    probs = torch.sigmoid(torch.tensor(logits[:, 1])).numpy()
    best_f2, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fb = f_beta(p, r)
        if fb > best_f2:
            best_f2, best_t = fb, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ConversationDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # token_type_ids: DeBERTa does not use them but some tokenizer versions return them
        tti = enc.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long))
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": tti.squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Class weights
# w_pos = 3.0 * (N_neg / N_pos),  w_neg = 1.0
# ---------------------------------------------------------------------------

def make_class_weights(labels: list, device: torch.device) -> torch.Tensor:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("No positive samples in training set.")
    w_pos = 3.0 * (n_neg / n_pos)
    print(f"  Class weights: neg=1.0  pos={w_pos:.2f}  (N_neg={n_neg}  N_pos={n_pos})")
    return torch.tensor([1.0, w_pos], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        label          = batch["label"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

        loss = loss_fn(logits, label) / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if step % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # flush remaining gradients if steps not divisible by grad_accum
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        logits = model(
            input_ids=      batch["input_ids"].to(device),
            attention_mask= batch["attention_mask"].to(device),
            token_type_ids= batch["token_type_ids"].to(device),
        ).logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["label"].numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return compute_metrics(logits, labels, threshold), logits, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH      = "pan12_dataset"
MODEL_NAME        = "microsoft/deberta-v3-base"
OUTPUT_DIR        = "runs/pan12"
EPOCHS            = 5
LR                = 2e-5
BATCH_SIZE        = 16
MAX_LENGTH        = 512
WARMUP_RATIO      = 0.10
WEIGHT_DECAY      = 0.01
GRAD_CLIP         = 1.0
GRAD_ACCUM        = 2
PATIENCE          = 2
# ---------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(out / "best")

    # --- Load data ---
    def load_split(name: str) -> tuple:
        path = Path(DATASET_PATH) / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        texts  = [r["text"]  for r in records]
        labels = [r["label"] for r in records]
        return texts, labels

    train_texts, train_labels = load_split("train")
    val_texts,   val_labels   = load_split("val")
    test_texts,  test_labels  = load_split("test")

    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

    # --- Model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # --- Loaders ---
    train_ds = ConversationDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds   = ConversationDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Loss ---
    weights = make_class_weights(train_labels, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )
    total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- Training loop ---
    best_f2, best_thresh, no_improve = 0.0, 0.5, 0

    print(f"\n--- PAN 2012 fine-tuning ({EPOCHS} epochs max) ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, GRAD_CLIP, GRAD_ACCUM,
        )

        _, val_logits, val_labels_np = evaluate(model, val_loader, device)
        thresh = calibrate_threshold(val_logits, val_labels_np)
        m = compute_metrics(val_logits, val_labels_np, thresh)

        print(
            f"  epoch {epoch:2d} | loss {train_loss:.4f} | "
            f"P {m['precision']:.3f} | R {m['recall']:.3f} | "
            f"F2 {m['f2']:.3f} | thresh {thresh:.2f} | "
            f"TP {m['tp']} FP {m['fp']} FN {m['fn']}"
        )

        if m["f2"] > best_f2:
            best_f2, best_thresh, no_improve = m["f2"], thresh, 0
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            with open(out / "best_threshold.json", "w", encoding="utf-8") as f:
                json.dump({"threshold": best_thresh, "f2": best_f2}, f, indent=2)
            print(f"    -> checkpoint saved (F2={best_f2:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    # --- Test evaluation ---
    print("\n--- Test set evaluation ---")
    model_test = AutoModelForSequenceClassification.from_pretrained(best_ckpt).to(device)
    tok_test   = AutoTokenizer.from_pretrained(best_ckpt)

    test_ds = ConversationDataset(
        test_texts, test_labels, tok_test, MAX_LENGTH
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    test_m, _, _ = evaluate(model_test, test_loader, device, threshold=best_thresh)
    print(
        f"  P={test_m['precision']:.3f}  R={test_m['recall']:.3f}  "
        f"F2={test_m['f2']:.3f}  threshold={best_thresh:.2f}  "
        f"TP={test_m['tp']}  FP={test_m['fp']}  FN={test_m['fn']}"
    )

    with open(out / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)
    print(f"  Saved to {out}/test_results.json")


if __name__ == "__main__":
    main()
