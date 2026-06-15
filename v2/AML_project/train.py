"""
pretrain.py
------------------------
Continued domain pretraining for DeBERTa with masked language modeling.

Data policy:
  train.json -> MLM training
  val.json   -> MLM validation
  test.json  -> not used

Run:
  python pretrain.py
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass(frozen=True)
class Config:
    dataset_path: str = "pan12_dataset"
    base_model: str = "microsoft/deberta-v3-base"
    output_dir: str = "runs/deberta_mlm"
    train_split: str = "train"
    eval_split: str = "val"

    max_length: int = 512
    min_tokens: int = 32
    stride: int = 64

    epochs: float = 3.0
    lr: float = 5e-5
    batch_size: int = 4
    eval_batch_size: int = 4
    grad_accum: int = 8
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    mlm_probability: float = 0.15
    max_special_tokens: int = 512

    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    num_workers: int = 2
    fp16: bool = True
    seed: int = 42


CFG = Config()
BRACKET_TOKEN_RE = re.compile(r"\[[^\]\s]+\]")


def read_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")
    return records


def load_texts(dataset_path: str, split: str) -> list[str]:
    path = Path(dataset_path) / f"{split}.json"
    records = read_json_records(path)
    texts = []
    for row in records:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def collect_special_tokens(texts: list[str], max_tokens: int) -> list[str]:
    tokens = set()
    for text in texts:
        tokens.update(BRACKET_TOKEN_RE.findall(text))
    ordered = sorted(tokens)
    if max_tokens > 0:
        ordered = ordered[:max_tokens]
    return ordered


def add_domain_tokens(tokenizer, texts: list[str], max_tokens: int) -> int:
    tokens = collect_special_tokens(texts, max_tokens)
    if not tokens:
        return 0
    return tokenizer.add_special_tokens({"additional_special_tokens": tokens})


def make_chunks(
    text: str,
    tokenizer,
    max_length: int,
    min_tokens: int,
    stride: int,
) -> list[dict]:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]

    special = tokenizer.num_special_tokens_to_add(pair=False)
    usable = max_length - special
    if usable <= 8:
        raise ValueError("max_length is too small for this tokenizer")

    step = usable - stride
    if step <= 0:
        raise ValueError("stride must be smaller than max_length minus special tokens")

    examples = []
    for start in range(0, len(ids), step):
        chunk = ids[start:start + usable]
        if len(chunk) < min_tokens:
            continue
        enc = tokenizer.prepare_for_model(
            chunk,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        examples.append(enc)
        if start + usable >= len(ids):
            break
    return examples


class MlmDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int,
        min_tokens: int,
        stride: int,
    ) -> None:
        self.examples = []
        for text in texts:
            self.examples.extend(
                make_chunks(text, tokenizer, max_length, min_tokens, stride)
            )
        if not self.examples:
            raise ValueError("No MLM examples were created")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def build_training_args(cfg: Config, total_steps: int):
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    common = dict(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        eval_steps=cfg.eval_steps,
        report_to="none",
        fp16=cfg.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=cfg.num_workers,
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    try:
        return TrainingArguments(evaluation_strategy="steps", **common)
    except TypeError:
        return TrainingArguments(eval_strategy="steps", **common)


def main() -> None:
    cfg = CFG
    set_seed(cfg.seed)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_texts = load_texts(cfg.dataset_path, cfg.train_split)
    eval_texts = load_texts(cfg.dataset_path, cfg.eval_split)

    print(f"base_model: {cfg.base_model}", flush=True)
    print(f"train_texts: {len(train_texts)}", flush=True)
    print(f"eval_texts: {len(eval_texts)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    added = add_domain_tokens(tokenizer, train_texts, cfg.max_special_tokens)
    print(f"added_special_tokens: {added}", flush=True)

    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
    if added:
        model.resize_token_embeddings(len(tokenizer))

    train_ds = MlmDataset(
        train_texts,
        tokenizer,
        cfg.max_length,
        cfg.min_tokens,
        cfg.stride,
    )
    eval_ds = MlmDataset(
        eval_texts,
        tokenizer,
        cfg.max_length,
        cfg.min_tokens,
        cfg.stride,
    )

    print(f"train_mlm_examples: {len(train_ds)}", flush=True)
    print(f"eval_mlm_examples: {len(eval_ds)}", flush=True)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
    )

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    targs = build_training_args(cfg, total_steps)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics, flush=True)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    config_dump = asdict(cfg)
    config_dump["added_special_tokens"] = added
    with (out / "pretrain_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    print(f"saved: {cfg.output_dir}", flush=True)


if __name__ == "__main__":
    main()
