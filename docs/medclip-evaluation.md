# MedCLIP image-text alignment evaluation

Run this after RoentGen inference has produced one synthetic image for each
held-out prompt. It measures whether the generated X-ray matches the same
BioGPT prompt more closely than a shuffled, incorrect prompt.

Use a fresh Colab runtime. The official MedCLIP package requires
`transformers<=4.24.0`, whose required `tokenizers` package has no Python 3.12
wheel. The supplied requirements keep a current Colab-compatible Transformers
version and the evaluator reproduces MedCLIP's image preprocessing directly.
Save all model and image outputs to Drive before starting this evaluation
runtime.

## Inputs

The evaluator accepts either a directory of text prompts or the original
predictions CSV. For direct image/label-folder evaluation, use:

```text
/content/xray-database/test/labels/
  3.txt, 7.txt, ...

/content/results/predicted/
  3.png, 7.png, ...
```

Images written by this repository's inference script as `3_0.jpg`, `7_0.jpg`,
and so on are also matched automatically to `3.txt`, `7.txt`. If both `3.jpg`
and `3_0.jpg` exist, remove the unintended file so the pair is unambiguous.

The original CSV mode remains available:

```text
/content/test_predictions.csv
  required columns: folder_stem, prediction

/content/drive/MyDrive/Projects/data/xray/roentgen_lora_v1/test_generated/
  3.png, 7.png, ... matching each folder_stem
```

The generated image directory must contain one matching image for every prompt
file or CSV row. The script stops on duplicate or ambiguous IDs, missing images,
blank prompts, or bad CSV column names instead of producing a partial score.

## Fresh Colab runtime

```bash
git clone https://github.com/mertz1999/RoentGen-v2.git
cd RoentGen-v2
bash scripts/install_medclip_colab.sh
```

For `/content/results/predicted` images and the test label folder, run:

```bash
python roentgenv2/evaluation/medclip_alignment.py \
  --prompt-dir /content/xray-database/test/labels \
  --generated-image-dir /content/results/predicted \
  --output-dir /content/medclip_evaluation \
  --batch-size 8 \
  --negative-shuffles 5 \
  --seed 873
```

To evaluate prompts stored in the predictions CSV instead, run:

```bash
python roentgenv2/evaluation/medclip_alignment.py \
  --predictions-csv /content/test_predictions.csv \
  --generated-image-dir /content/drive/MyDrive/Projects/data/xray/roentgen_lora_v1/test_generated \
  --output-dir /content/drive/MyDrive/Projects/data/xray/roentgen_lora_v1/medclip_evaluation \
  --batch-size 8 \
  --negative-shuffles 5 \
  --seed 873
```

The first run downloads the MedCLIP-ViT weights. Reduce `--batch-size` to `4`
if the GPU runs out of memory.

If an older clone is already present, update it before installing:

```bash
cd /content/RoentGen-v2
git pull origin main
```

## Outputs

```text
medclip_evaluation/
  alignment_per_pair.csv        every matched and negative score
  alignment_summary.json        means, standard deviations, bootstrap CIs, AUROC, retrieval
  alignment_distribution.png    matched versus negative score distributions
```

Interpretation:

- `matched.mean` should be higher than `negative.mean`.
- The confidence interval for `matched_minus_negative.mean` should be above
  zero, and the one-sided permutation p-value should be small.
- `matched_vs_negative_auroc` should be above 0.50; higher is better.
- Do not combine the raw MedCLIP value with raw FID. FID is a global,
  unbounded distribution score and must be normalized separately for the final
  HybridScore.
