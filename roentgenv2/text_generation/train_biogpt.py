#!/usr/bin/env python3
"""Fine-tune BioGPT to produce standardized RoentGen conditioning prompts."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class PromptDataset(Dataset):
    """Causal-LM examples that calculate loss only over the target prompt."""

    def __init__(self, jsonl_file: Path, tokenizer: Any, max_source_length: int, max_target_length: int):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        with jsonl_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                data = json.loads(line)
                if not data.get("source") or not data.get("target"):
                    raise ValueError(f"{jsonl_file}:{line_number} must have non-empty source and target")
                self.examples.append(data)
        if not self.examples:
            raise ValueError(f"No usable examples in {jsonl_file}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        source_ids = self.tokenizer.encode(
            example["source"], add_special_tokens=True, truncation=True, max_length=self.max_source_length
        )
        target_ids = self.tokenizer.encode(
            example["target"], add_special_tokens=False, truncation=True, max_length=self.max_target_length - 1
        )
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        input_ids = source_ids + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(source_ids) + target_ids,
        }


@dataclass
class CausalPaddingCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class MetricsCallback(TrainerCallback):
    """Persist useful loss history after logs, validation, checkpoints, and end."""

    def __init__(self, metrics_path: Path, metadata: dict[str, Any]):
        self.metrics_path = metrics_path
        self.payload = read_json(metrics_path, {"meta": metadata, "train": [], "validation": [], "test": {}, "final": {}})
        self.payload["meta"] = {**self.payload.get("meta", {}), **metadata}

    @staticmethod
    def _upsert(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
        step = record.get("step")
        for position, existing in enumerate(records):
            if existing.get("step") == step:
                records[position] = {**existing, **record}
                return
        records.append(record)

    def _save(self) -> None:
        write_json_atomic(self.metrics_path, self.payload)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" in logs:
            self._upsert(
                self.payload.setdefault("train", []),
                {
                    "step": state.global_step,
                    "epoch": logs.get("epoch"),
                    "loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                },
            )
        self._save()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        prefix = "test" if any(name.startswith("test_") for name in metrics) else "validation"
        cleaned = {name: value for name, value in metrics.items() if isinstance(value, (int, float))}
        if prefix == "test":
            self.payload["test"] = {"step": state.global_step, **cleaned}
        else:
            self._upsert(self.payload.setdefault("validation", []), {"step": state.global_step, **cleaned})
        self._save()
        return control

    def on_save(self, args, state, control, **kwargs):
        self._save()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save()
        return control


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for candidate in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(candidate.name.rsplit("-", 1)[1]), candidate))
        except ValueError:
            continue
    return str(max(checkpoints)[1]) if checkpoints else None


def finite_perplexity(loss: float | None) -> float | None:
    if loss is None or not math.isfinite(loss):
        return None
    return float(math.exp(min(loss, 20)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", default=None, help="override config: latest, a checkpoint path, or none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config_file.read_text(encoding="utf-8")) or {}
    required = {"model_name_or_path", "train_file", "validation_file", "test_file", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(sorted(missing))}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_key in ("train_file", "validation_file", "test_file"):
        if not Path(config[file_key]).is_file():
            raise FileNotFoundError(f"{file_key} does not exist: {config[file_key]}")

    seed = int(config.get("seed", 873))
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"])
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    max_source_length = int(config.get("max_source_length", 384))
    max_target_length = int(config.get("max_target_length", 128))
    datasets = {
        "train": PromptDataset(Path(config["train_file"]), tokenizer, max_source_length, max_target_length),
        "validation": PromptDataset(Path(config["validation_file"]), tokenizer, max_source_length, max_target_length),
        "test": PromptDataset(Path(config["test_file"]), tokenizer, max_source_length, max_target_length),
    }

    requested_fp16 = bool(config.get("fp16", True))
    fp16 = requested_fp16 and torch.cuda.is_available()
    if requested_fp16 and not fp16:
        print("WARNING: fp16 requested but CUDA is unavailable; training will be extremely slow on CPU.")

    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        logging_strategy="steps",
        logging_steps=int(config.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(config.get("save_steps", 100)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        report_to=config.get("report_to", "none"),
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    evaluation_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    kwargs[evaluation_key] = "steps"
    kwargs["eval_steps"] = int(config.get("eval_steps", 100))
    training_args = TrainingArguments(**kwargs)

    metrics_path = output_dir / config.get("metrics_file", "metrics.json")
    metadata = {
        "model_name_or_path": config["model_name_or_path"],
        "seed": seed,
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "train_examples": len(datasets["train"]),
        "validation_examples": len(datasets["validation"]),
        "test_examples": len(datasets["test"]),
        "config_file": str(args.config_file.resolve()),
    }
    metrics_callback = MetricsCallback(metrics_path, metadata)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CausalPaddingCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    desired_resume = args.resume_from_checkpoint if args.resume_from_checkpoint is not None else config.get("resume_from_checkpoint", "latest")
    if desired_resume == "latest":
        resume_checkpoint = latest_checkpoint(output_dir)
    elif desired_resume in (None, "none", "None", ""):
        resume_checkpoint = None
    else:
        resume_checkpoint = str(desired_resume)
        if not Path(resume_checkpoint).is_dir():
            raise FileNotFoundError(f"Requested checkpoint does not exist: {resume_checkpoint}")
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    validation_metrics = trainer.evaluate(metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
    validation_metrics["eval_perplexity"] = finite_perplexity(validation_metrics.get("eval_loss"))
    test_metrics["test_perplexity"] = finite_perplexity(test_metrics.get("test_loss"))

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    shutil.copy2(args.config_file, output_dir / "training_config.yaml")

    metrics_callback.payload["test"] = {"step": trainer.state.global_step, **test_metrics}
    metrics_callback.payload["final"] = {
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "train": {name: value for name, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "validation": {name: value for name, value in validation_metrics.items() if isinstance(value, (int, float))},
        "test": {name: value for name, value in test_metrics.items() if isinstance(value, (int, float))},
        "final_model_dir": str(final_model_dir),
    }
    metrics_callback._save()
    print(json.dumps(metrics_callback.payload["final"], indent=2))


if __name__ == "__main__":
    main()
