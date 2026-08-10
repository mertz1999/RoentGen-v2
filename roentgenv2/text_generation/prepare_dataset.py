#!/usr/bin/env python3
"""Build leakage-checked BioGPT splits from the Claude prompt CSV.

The existing RoentGen image folders define the external train/test split.  This
script maps their label/image stems to ``report_id`` values in the CSV, creates
a validation subset only from the external train split, and writes JSONL files
for causal language-model training.

Important: ``generated_prompt`` is made from the source report.  Consequently,
this is a *report-to-standardized-prompt distillation* task, not an independent
clinical report generation task.  The default inputs deliberately exclude the
target, identifiers, image names, token counts, costs, and error fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_INPUT_FIELDS = ["indication", "findings", "impression", "mesh_labels", "num_images"]
ALLOWED_INPUT_FIELDS = {
    "indication",
    "findings",
    "impression",
    "mesh_labels",
    "num_images",
    # ``full_report`` is deliberately opt-in: it duplicates the other report
    # sections and makes this a more direct report-rewriting task.
    "full_report",
}


def normalise_identifier(value: object) -> str:
    """Return an upper-case alphanumeric comparison key."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def report_key_from_stem(stem: str) -> str:
    """Map common IU-X-Ray filename forms to a report key.

    Examples: ``1002``, ``CXR1002`` and ``CXR1002_IM-1234-1001`` all map to
    ``CXR1002``.  Unknown stems stay unchanged and later fail validation rather
    than silently being assigned to an arbitrary report.
    """
    raw = str(stem).strip().upper()
    # Keep separators here: ``CXR1_1_IM-0001-3001`` belongs to report CXR1,
    # whereas normalising first would incorrectly turn it into CXR11.
    match = re.match(r"^(?:CXR)?0*(\d+)(?:(?:[_-](?:\d+|IM).*)|(?:IM.*))?$", raw)
    if match:
        return f"CXR{int(match.group(1))}"
    return normalise_identifier(raw)


def clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_source_text(row: dict[str, object], input_fields: Iterable[str]) -> str:
    """Create the input side of the supervised example from clinical fields."""
    labels = {
        "indication": "Indication",
        "findings": "Findings",
        "impression": "Impression",
        "mesh_labels": "Structured labels",
        "num_images": "Number of images",
        "full_report": "Source report",
    }
    sections = []
    for field in input_fields:
        value = clean_value(row.get(field))
        if value:
            sections.append(f"{labels[field]}: {value}")
    if not sections:
        raise ValueError("row has no usable configured clinical input fields")
    return "Clinical information:\n" + "\n".join(sections) + "\n\nStandardized radiology prompt:\n"


def discover_folder_stems(split_dir: Path) -> tuple[set[str], list[str]]:
    """Return usable label stems and data-integrity warnings for one split."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected both {images_dir} and {labels_dir}. "
            "Pass the xray-database directory, not its train/test subfolder."
        )

    image_stems = {path.stem for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}
    warnings = []
    missing_images = sorted(label_stems - image_stems)
    missing_labels = sorted(image_stems - label_stems)
    if missing_images:
        warnings.append(f"{split_dir.name}: {len(missing_images)} labels have no matching image")
    if missing_labels:
        warnings.append(f"{split_dir.name}: {len(missing_labels)} images have no matching label")
    return label_stems & image_stems, warnings


def index_csv_rows(frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    required = {"report_id", "generated_prompt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    index: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        report_id = clean_value(row["report_id"])
        target = clean_value(row["generated_prompt"])
        if report_id and target:
            index[report_key_from_stem(report_id)].append(row)
    return index


def map_stems_to_rows(stems: Iterable[str], row_index: dict[str, list[dict[str, object]]], split: str) -> list[dict[str, object]]:
    """Map folder stems to exactly one CSV row; fail on ambiguity or missing IDs."""
    examples = []
    missing, ambiguous = [], []
    seen_report_ids = set()
    for stem in sorted(stems, key=lambda value: (len(value), value)):
        matches = row_index.get(report_key_from_stem(stem), [])
        if not matches:
            missing.append(stem)
            continue
        if len(matches) != 1:
            ambiguous.append(stem)
            continue
        row = matches[0]
        report_id = clean_value(row["report_id"])
        # Several image files from the same study must stay one text example.
        if report_id in seen_report_ids:
            continue
        seen_report_ids.add(report_id)
        examples.append({"report_id": report_id, "folder_stem": stem, "split": split, "row": row})
    if missing or ambiguous:
        details = []
        if missing:
            details.append(f"{len(missing)} unmapped stems (examples: {', '.join(missing[:10])})")
        if ambiguous:
            details.append(f"{len(ambiguous)} ambiguous stems (examples: {', '.join(ambiguous[:10])})")
        raise ValueError(f"Could not map {split} folder files to CSV report_id: {'; '.join(details)}")
    return examples


def deterministic_validation_split(examples: list[dict[str, object]], fraction: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0 < fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if len(examples) < 2:
        raise ValueError("Need at least two external-train examples to create train and validation splits")
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def write_jsonl(path: Path, examples: list[dict[str, object]], input_fields: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in examples:
            row = item["row"]
            payload = {
                "report_id": item["report_id"],
                "folder_stem": item["folder_stem"],
                "source": build_source_text(row, input_fields),
                "target": clean_value(row["generated_prompt"]),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def label_match_stats(examples: Iterable[dict[str, object]], labels_dir: Path) -> dict[str, int]:
    """Check labels only as an audit.  CSV targets remain canonical."""
    stats: Counter[str] = Counter()
    for item in examples:
        label_file = labels_dir / f"{item['folder_stem']}.txt"
        if not label_file.exists():
            stats["missing"] += 1
            continue
        label = " ".join(label_file.read_text(encoding="utf-8", errors="replace").split())
        target = clean_value(item["row"]["generated_prompt"])
        stats["exact_match" if label == target else "different"] += 1
    return dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", type=Path, required=True, help="IU X-Ray Claude-prompt CSV")
    parser.add_argument("--xray-database", type=Path, required=True, help="contains train/ and test/ image+label folders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-fields", nargs="+", default=DEFAULT_INPUT_FIELDS, choices=sorted(ALLOWED_INPUT_FIELDS))
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=873)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_file.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv_file}")
    frame = pd.read_csv(args.csv_file)
    row_index = index_csv_rows(frame)

    train_stems, train_warnings = discover_folder_stems(args.xray_database / "train")
    test_stems, test_warnings = discover_folder_stems(args.xray_database / "test")
    train_examples = map_stems_to_rows(train_stems, row_index, "external_train")
    test_examples = map_stems_to_rows(test_stems, row_index, "external_test")

    train_ids = {item["report_id"] for item in train_examples}
    test_ids = {item["report_id"] for item in test_examples}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"External train/test leakage: {len(overlap)} report_id values occur in both folders "
            f"(examples: {', '.join(sorted(overlap)[:10])})"
        )

    fit_examples, validation_examples = deterministic_validation_split(train_examples, args.validation_fraction, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", fit_examples, args.input_fields)
    write_jsonl(args.output_dir / "validation.jsonl", validation_examples, args.input_fields)
    write_jsonl(args.output_dir / "test.jsonl", test_examples, args.input_fields)

    manifest_rows = []
    for subset, examples in (("train", fit_examples), ("validation", validation_examples), ("test", test_examples)):
        manifest_rows.extend({"subset": subset, "report_id": item["report_id"], "folder_stem": item["folder_stem"]} for item in examples)
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subset", "report_id", "folder_stem"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "csv_file": str(args.csv_file.resolve()),
        "xray_database": str(args.xray_database.resolve()),
        "input_fields": args.input_fields,
        "target_field": "generated_prompt",
        "task_definition": "clinical-report-to-standardized-prompt distillation",
        "split_policy": "external train/test folders are preserved; validation is sampled only from external train",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "examples": {"train": len(fit_examples), "validation": len(validation_examples), "test": len(test_examples)},
        "folder_warnings": train_warnings + test_warnings,
        "label_audit": {
            "external_train": label_match_stats(train_examples, args.xray_database / "train" / "labels"),
            "external_test": label_match_stats(test_examples, args.xray_database / "test" / "labels"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
