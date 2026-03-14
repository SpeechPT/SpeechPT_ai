"""CE LoRA training script.

Input JSONL format:
{"text_a": "...", "text_b": "...", "label": 0}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float((preds == labels).mean())
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"accuracy": acc, "f1": float(f1), "precision": float(precision), "recall": float(recall)}


def main():
    parser = argparse.ArgumentParser(description="Train CE pair classifier with LoRA")
    parser.add_argument("--model", default="klue/roberta-base")
    parser.add_argument("--train", required=True, help="train jsonl path")
    parser.add_argument("--valid", required=True, help="valid jsonl path")
    parser.add_argument("--output", default="artifacts/ce_model")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing training dependencies. Install with `pip install datasets peft accelerate transformers`."
        ) from exc

    train_rows = load_jsonl(Path(args.train))
    valid_rows = load_jsonl(Path(args.valid))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(example):
        encoded = tokenizer(
            example["text_a"],
            example["text_b"],
            truncation=True,
            max_length=args.max_length,
        )
        encoded["labels"] = int(example["label"])
        return encoded

    train_ds = Dataset.from_list(train_rows).map(preprocess, remove_columns=list(train_rows[0].keys()))
    valid_ds = Dataset.from_list(valid_rows).map(preprocess, remove_columns=list(valid_rows[0].keys()))

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
