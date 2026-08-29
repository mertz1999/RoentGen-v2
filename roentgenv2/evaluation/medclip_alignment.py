#!/usr/bin/env python3
"""Measure matched image/prompt similarity with MedCLIP.

Run this only in the separate MedCLIP environment described in
``docs/medclip-evaluation.md``. Each image is scored only against its matching
prompt; this evaluator does not construct or score negative pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MEDCLIP_REVISION = "9c3396f20d5d54e4fae241b8cb06ca45848e98c9"
MEDCLIP_IMAGE_SIZE = 224
MEDCLIP_IMAGE_MEAN = 0.5862785803043838
MEDCLIP_IMAGE_STD = 0.27950088968644304
MEDCLIP_TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stem_from_value(value: object) -> str:
    """Keep numeric CSV IDs consistent with filenames such as ``1002.png``."""
    if pd.isna(value):
        raise ValueError("found an empty image-ID value")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def image_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Generated-image directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    duplicates = set()
    index: dict[str, Path] = {}
    for path in paths:
        if path.stem in index:
            duplicates.add(path.stem)
        index[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate generated image stems: {', '.join(sorted(duplicates)[:10])}")
    if not index:
        raise ValueError(f"No PNG/JPG/JPEG files found in {directory}")
    return index


def load_csv_pairs(
    csv_path: Path,
    generated_dir: Path,
    id_column: str,
    prompt_column: str,
) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing_columns = {id_column, prompt_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if frame[id_column].isna().any() or frame[prompt_column].isna().any():
        raise ValueError("CSV contains empty image IDs or prompts")

    frame = frame.copy()
    frame["folder_stem"] = frame[id_column].map(stem_from_value)
    frame["prompt"] = frame[prompt_column].astype(str).str.strip()
    if (frame["prompt"] == "").any():
        raise ValueError("CSV contains blank prompts")
    if frame["folder_stem"].duplicated().any():
        duplicates = frame.loc[frame["folder_stem"].duplicated(), "folder_stem"].head(10).tolist()
        raise ValueError(f"CSV has duplicate image IDs: {duplicates}")

    images = image_index(generated_dir)
    frame["image_path"] = frame["folder_stem"].map(images)
    missing_images = frame.loc[frame["image_path"].isna(), "folder_stem"].head(10).tolist()
    if missing_images:
        raise ValueError(f"No generated image matches these CSV IDs: {missing_images}")
    return frame.reset_index(drop=True)


def load_prompt_dir_pairs(prompt_dir: Path, generated_dir: Path) -> pd.DataFrame:
    """Pair ``<id>.txt`` prompts with ``<id>`` or ``<id>_0`` generated images."""

    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory does not exist: {prompt_dir}")
    prompt_paths = sorted(
        path for path in prompt_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not prompt_paths:
        raise ValueError(f"No TXT prompt files found in {prompt_dir}")

    images = image_index(generated_dir)
    rows = []
    missing_images = []
    for prompt_path in prompt_paths:
        stem = prompt_path.stem
        candidates = [
            image_path
            for image_stem in (stem, f"{stem}_0")
            if (image_path := images.get(image_stem)) is not None
        ]
        if not candidates:
            missing_images.append(stem)
            continue
        if len(candidates) > 1:
            candidate_names = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"Multiple generated images match prompt {prompt_path.name}: {candidate_names}"
            )

        prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is blank: {prompt_path}")
        rows.append(
            {
                "folder_stem": stem,
                "prompt_path": prompt_path,
                "prompt": prompt,
                "image_path": candidates[0],
            }
        )

    if missing_images:
        raise ValueError(
            "No generated image matches these prompt files: "
            f"{missing_images[:10]}"
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    means = np.empty(samples, dtype=np.float64)
    count = len(values)
    for index in range(samples):
        means[index] = values[rng.integers(0, count, size=count)].mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def paired_similarities(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    logit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw cosine and official temperature-scaled logits per pair."""

    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            "Image and text embedding arrays must have the same shape; "
            f"got {image_embeddings.shape} and {text_embeddings.shape}."
        )
    raw_cosine = np.sum(image_embeddings * text_embeddings, axis=1)
    return raw_cosine, raw_cosine * float(logit_scale)


def preprocess_medclip_image(image: "Image.Image") -> np.ndarray:
    """Reproduce MedCLIP's documented grayscale square-pad/224 normalization.

    The original MedCLIP processor relies on an old Transformers image API.
    Keeping this small preprocessing implementation here lets the same MedCLIP
    weights run in current Python 3.12 Colab without downgrading tokenizers.
    """
    from PIL import Image

    width, height = image.size
    size = max(MEDCLIP_IMAGE_SIZE, width, height)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(image, ((size - width) // 2, (size - height) // 2))
    canvas = canvas.resize((MEDCLIP_IMAGE_SIZE, MEDCLIP_IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    pixels = (pixels - MEDCLIP_IMAGE_MEAN) / MEDCLIP_IMAGE_STD
    return pixels[None, :, :]


def load_medclip_vit_weights(model: object, checkpoint_dir: Path) -> None:
    """Load official MedCLIP weights across old/new Transformers buffer changes.

    Modern Transformers does not persist BioClinicalBERT's ``position_ids``
    buffer. The original MedCLIP checkpoint includes it, so strict loading
    fails even though every learned parameter matches. Only that one known
    obsolete buffer is permitted in the fallback; any learned-weight mismatch
    still stops evaluation.
    """
    import torch

    try:
        model.from_pretrained(input_dir=str(checkpoint_dir))
        return
    except RuntimeError as error:
        state_path = checkpoint_dir / "pytorch_model.bin"
        if not state_path.is_file():
            raise RuntimeError(
                "MedCLIP checkpoint loading failed before the checkpoint was available. "
                "Delete the incomplete checkpoint directory and rerun the command."
            ) from error
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        incompatibility = model.load_state_dict(state_dict, strict=False)
        unexpected = set(incompatibility.unexpected_keys)
        missing = set(incompatibility.missing_keys)
        allowed_unexpected = {"text_model.model.embeddings.position_ids"}
        if missing or unexpected - allowed_unexpected:
            raise RuntimeError(
                "MedCLIP checkpoint has an unexpected learned-weight mismatch. "
                f"Missing: {sorted(missing)}; unexpected: {sorted(unexpected)}"
            ) from error
        print("Loaded MedCLIP-ViT with the legacy position_ids buffer safely ignored.")


def encode_pairs(
    frame: pd.DataFrame,
    batch_size: int,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("MedCLIP's reference implementation requires a CUDA GPU. Enable a Colab GPU runtime.")
    try:
        from medclip import MedCLIPModel, MedCLIPVisionModelViT
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - depends on isolated Colab environment
        raise RuntimeError(
            "MedCLIP is not installed. Use a fresh Colab runtime and install "
            "requirements-medclip.txt before running this script."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(MEDCLIP_TEXT_MODEL)
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    load_medclip_vit_weights(model, checkpoint_dir)
    model.cuda().eval()
    logit_scale = float(model.logit_scale.detach().clamp(0, 4.6052).exp().cpu().item())

    image_embeddings: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            images = []
            for path in batch["image_path"]:
                with Image.open(path) as image:
                    images.append(preprocess_medclip_image(image))
            text_inputs = tokenizer(
                batch["prompt"].tolist(),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            outputs = model(
                pixel_values=torch.tensor(np.stack(images), dtype=torch.float32),
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            image_embeddings.append(outputs["img_embeds"].detach().cpu().numpy())
            text_embeddings.append(outputs["text_embeds"].detach().cpu().numpy())
    return np.concatenate(image_embeddings), np.concatenate(text_embeddings), logit_scale


def plot_similarity_distributions(
    raw_cosine: np.ndarray,
    scaled_logits: np.ndarray,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(raw_cosine, bins=30, alpha=0.75, color="tab:blue")
    axes[0].axvline(raw_cosine.mean(), color="black", linestyle="--", label="mean")
    axes[0].set_xlabel("raw cosine similarity")
    axes[0].set_ylabel("number of matched pairs")
    axes[0].set_title("Matched-pair cosine")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(scaled_logits, bins=30, alpha=0.75, color="tab:green")
    axes[1].axvline(scaled_logits.mean(), color="black", linestyle="--", label="mean")
    axes[1].set_xlabel("temperature-scaled MedCLIP logit")
    axes[1].set_ylabel("number of matched pairs")
    axes[1].set_title("Matched-pair scaled logit")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def similarity_summary(values: np.ndarray, mean_ci: tuple[float, float]) -> dict[str, object]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1 if len(values) > 1 else 0)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "mean_95_bootstrap_ci": list(mean_ci),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--predictions-csv",
        type=Path,
        help="CSV containing image IDs and prompts",
    )
    input_group.add_argument(
        "--prompt-dir",
        "--label-dir",
        dest="prompt_dir",
        type=Path,
        help="Directory of <image-id>.txt prompts; accepts <image-id> or <image-id>_0 images",
    )
    parser.add_argument("--generated-image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default="folder_stem", help="CSV column matching generated image stems")
    parser.add_argument("--prompt-column", default="prediction", help="CSV text column used to generate images")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=873)
    parser.add_argument(
        "--medclip-checkpoint-dir",
        type=Path,
        default=Path("pretrained/medclip-vit"),
        help="Cache directory for official MedCLIP-ViT weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("--batch-size and --bootstrap-samples must be positive")
    if args.predictions_csv is not None:
        frame = load_csv_pairs(
            args.predictions_csv,
            args.generated_image_dir,
            args.id_column,
            args.prompt_column,
        )
        input_summary = {
            "type": "predictions_csv",
            "path": str(args.predictions_csv.resolve()),
            "sha256": sha256(args.predictions_csv),
        }
    else:
        frame = load_prompt_dir_pairs(args.prompt_dir, args.generated_image_dir)
        input_summary = {
            "type": "prompt_directory",
            "path": str(args.prompt_dir.resolve()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings, text_embeddings, logit_scale = encode_pairs(
        frame,
        args.batch_size,
        args.medclip_checkpoint_dir,
    )
    raw_cosine, scaled_logits = paired_similarities(
        image_embeddings,
        text_embeddings,
        logit_scale,
    )
    ci_rng = np.random.default_rng(args.seed + 1)
    raw_cosine_ci = bootstrap_mean_ci(raw_cosine, ci_rng, args.bootstrap_samples)
    scaled_logit_ci = tuple(value * logit_scale for value in raw_cosine_ci)

    output = frame.copy()
    output["raw_cosine_similarity"] = raw_cosine
    output["scaled_medclip_logit"] = scaled_logits
    keep_columns = [
        column
        for column in [
            "report_id",
            "folder_stem",
            "image_path",
            "prompt_path",
            "prompt",
            "raw_cosine_similarity",
            "scaled_medclip_logit",
        ]
        if column in output.columns
    ]
    output[keep_columns].to_csv(args.output_dir / "similarity_per_pair.csv", index=False)
    plot_similarity_distributions(
        raw_cosine,
        scaled_logits,
        args.output_dir / "similarity_distribution.png",
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "MedCLIP similarity for matched image/prompt pairs only",
        "model": "MedCLIP-ViT",
        "medclip_git_revision": MEDCLIP_REVISION,
        "input_source": input_summary,
        "generated_image_dir": str(args.generated_image_dir.resolve()),
        "medclip_checkpoint_dir": str(args.medclip_checkpoint_dir.resolve()),
        "n_pairs": len(frame),
        "seed": args.seed,
        "temperature_scaling": {
            "logit_scale": logit_scale,
            "formula": "scaled_medclip_logit = raw_cosine_similarity * logit_scale",
            "note": "The scaled logit is not a probability and does not change pair ranking.",
        },
        "raw_cosine_similarity": similarity_summary(raw_cosine, raw_cosine_ci),
        "scaled_medclip_logit": similarity_summary(scaled_logits, scaled_logit_ci),
        "software": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "medclip": package_version("MedCLIP"),
            "numpy": package_version("numpy"),
        },
        "preprocessing": {
            "image": "grayscale, zero-padded square, bicubic resize to 224x224, MedCLIP normalization",
            "text": f"{MEDCLIP_TEXT_MODEL}, max_length=77",
        },
        "files": {
            "per_pair": "similarity_per_pair.csv",
            "distribution_figure": "similarity_distribution.png",
        },
        "limitations": [
            "Only matched image/prompt pairs are scored; no negative comparison is performed.",
            "Absolute cosine similarity and scaled logits have no universal pass/fail threshold.",
            "Use the same dataset and settings when comparing real, base-model, and fine-tuned images.",
        ],
    }
    if args.predictions_csv is not None:
        summary["predictions_csv"] = input_summary["path"]
        summary["predictions_csv_sha256"] = input_summary["sha256"]
    else:
        summary["prompt_dir"] = input_summary["path"]
    (args.output_dir / "similarity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
