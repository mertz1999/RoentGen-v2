#!/usr/bin/env python3
"""Generate standardized text prompts from a trained BioGPT checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="the output_dir/final_model directory")
    parser.add_argument("--input-file", type=Path, required=True, help="prepared train, validation, or test JSONL")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=384, help="must match the training configuration")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_examples(path: Path, limit: int | None) -> list[dict]:
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return examples[:limit] if limit is not None else examples


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    examples = load_examples(args.input_file, args.limit)
    if not examples:
        raise ValueError(f"No examples found in {args.input_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["report_id", "folder_stem", "source", "target", "prediction"])
        writer.writeheader()
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            encoded = tokenizer(
                [example["source"] for example in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            predictions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for example, prediction in zip(batch, predictions):
                writer.writerow({
                    "report_id": example["report_id"],
                    "folder_stem": example.get("folder_stem", ""),
                    "source": example["source"],
                    "target": example["target"],
                    "prediction": " ".join(prediction.split()),
                })
    print(f"Wrote {len(examples)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
