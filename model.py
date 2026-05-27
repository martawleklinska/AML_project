import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModel, get_linear_schedule_with_warmup)

# CONFIG
# ===================================
DATASET_PATH = "pan12_dataset"
OUTPUT_DIR = "results"
BASE_MODEL = "distilbert-base-uncased"

EPOCHS = 6
LR = 3E-05
BATCH_SIZE = 32
MAX_LENGTH = 256
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
GRAD_ACCUM = 1
PATIENCE = 3
BETA = 2.
DROPOUT = 0.3

# =============== METRICS
def f_beta(precision: float, recall: float, beta:float = BETA) -> float:
    denom = beta * beta * precision + recall
    if denom == 0.:
        return 0.
    return (1 + beta * beta) * precision * recall / denom

def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.

    return {"precision": precision,
            "recall": recall,
            "f2": f_beta(precision, recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "threshold": threshold,}

def calibrate_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    grid search: [0.05, 0.95] every 0.01 - we are looking for a threshold maximizing f2
    """
    best_f2, best_t = 0., 0.5

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


## =============DATA SET ===================

class GroomingDataset(Dataset):
    """
    Load records from jSON and tokenises the text.
    The tokens [RISK: HIGH], [ESC:1] go to the tokenizer and DistilBERT learns them from context
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
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
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float32),
        }
    
## =========== MODEL =============

class GroomingClassified(nn.Module):
    """
    DistilBERT + binar classifier on [CLS] token.
    [CLS] (768)-> Dropout(0.3) -> Linear(768->256) -> ReLU
                    Dropout(0.3) -> Linear(256->1) -> Sigmoid 
    """

    def __init__(self, base_model_name: str, dropout: float = DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        # outputs.last_hidden_state: (batch, seq_len, 768)
        # we take [CLS] token - idx 0
        cls_emb = outputs.last_hidden_state[:, 0, :] # (batch, 768)
        logit = self.classifier(cls_emb) # (batch, 1)
        return logit.squeeze(1)
    

# ==== CLASS WEIGHTS = compensation unbalanced dataset
def make_pos_weight(labels: list, device: torch.device) -> torch.Tensor:
    """
    BCEWithLogitsLoss takes pos_weight = N_neg / N_pos * scale
    Scale = 2. additionally makes the kara worse for missing grooming (FN)
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("Brak pozytywnych próbek w zbiorze treningowym!")
    pos_weight = 2.0 * (n_neg / n_pos)
    print(f"  Pos weight: {pos_weight:.2f}  (N_neg={n_neg}, N_pos={n_pos})")
    return torch.tensor([pos_weight], device=device)


# ==== TRAINING LOOP ==============
def run_epoch(
        model, loader, optimizer, scheduler, loss_fn, device, grad_clip, grad_accum
) -> float:
    model.train()
    total_loss = 0.
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask) # raw logity without sigmoid
        loss = loss_fn(logits, labels) / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if step % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
    # flush other grads
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, threshold = 0.5) -> tuple:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].numpy())
    
    probs = np.concatenate(all_probs)

    labels = np.concatenate(all_labels).astype(int)

    return compute_metrics(probs, labels, threshold), probs, labels

## =========== LOADING THE DATA =============
def load_split(dataset_path: str, name: str) -> tuple:
    """
    loads train.json / val.json / test/json
    every file is a list of records with "text" and "label"
    """
    path = Path(dataset_path) / f"{name}.json"
    with open(path, encoding = "utf-8") as f:
        records = json.load(f)

    texts  = [r["text"]  for r in records]
    labels = [int(r["label"]) for r in records]

    return texts, labels

## ======= main ===============

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device using: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents = True, exist_ok = True)

    # data
    print("\n loading data...")
    train_texts, train_labels = load_split(DATASET_PATH, "train")
    val_texts, val_labels = load_split(DATASET_PATH, "val")
    test_texts, test_labels = load_split(DATASET_PATH, "test")

    print(f"train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}")

    print(f"\n train positives: {sum(train_labels)} / {len(train_labels)}")
    print(f"({100*sum(train_labels)/len(train_labels):.1f}%)")

    # tokenizer and model
    print(f"\n loading model... {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = GroomingClassified(BASE_MODEL, DROPOUT).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params:,}")
    # ============== DataLoaders ===================
    train_ds = GroomingDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds   = GroomingDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=(device.type=="cuda"))

    # --- Loss z wagowaniem ---
    pos_weight = make_pos_weight(train_labels, device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # BCEWithLogitsLoss = Sigmoid + BCE — numerycznie stabilniejsze niż oddzielnie

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )
    total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n  Steps all: {total_steps}  Warmup: {warmup_steps}")

    # --- Pętla treningowa ---
    best_f2, best_thresh, no_improve = 0.0, 0.5, 0
    best_ckpt = str(out / "best_model.pt")

    print(f"\n--- Trening ({EPOCHS} epok max, early stopping patience={PATIENCE}) ---")
    print(f"{'Epoch':>5} | {'Loss':>7} | {'Prec':>6} | {'Rec':>6} | "
          f"{'F2':>6} | {'Thresh':>6} | TP  FP  FN")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, GRAD_CLIP, GRAD_ACCUM,
        )

        # Ewaluacja na val
        _, val_probs, val_labels_np = evaluate(model, val_loader, device)
        thresh = calibrate_threshold(val_probs, val_labels_np)
        m      = compute_metrics(val_probs, val_labels_np, thresh)

        print(
            f"{epoch:>5} | {train_loss:>7.4f} | {m['precision']:>6.3f} | "
            f"{m['recall']:>6.3f} | {m['f2']:>6.3f} | {thresh:>6.2f} | "
            f"{m['tp']:>3} {m['fp']:>3} {m['fn']:>3}"
        )

        if m["f2"] > best_f2:
            best_f2, best_thresh, no_improve = m["f2"], thresh, 0
            # Zapisujemy tylko wagi modelu (lżej niż cały checkpoint)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "base_model": BASE_MODEL,
                    "dropout": DROPOUT,
                    "max_length": MAX_LENGTH,
                },
            }, best_ckpt)
            with open(out / "best_threshold.json", "w") as f:
                json.dump({"threshold": best_thresh, "f2": best_f2}, f, indent=2)
            print(f"-> saved checkpoint (F2={best_f2:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping after epoch {epoch} (nothing better  {PATIENCE} epoch)")
                break

    # --- Ewaluacja on test ---
    print("\n--- test evaluation ---")

    # Wczytaj najlepszy model
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_ds     = GroomingDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    test_m, _, _ = evaluate(model, test_loader, device, threshold=best_thresh)
    print(
        f"  Precision : {test_m['precision']:.4f}\n"
        f"  Recall    : {test_m['recall']:.4f}\n"
        f"  F2        : {test_m['f2']:.4f}\n"
        f"  Threshold : {best_thresh:.2f}\n"
        f"  TP={test_m['tp']}  FP={test_m['fp']}  FN={test_m['fn']}"
    )

    with open(out / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"\n  results saved in {out}/test_results.json")
    print(f"model saved in {best_ckpt}")


if __name__ == "__main__":
    main()