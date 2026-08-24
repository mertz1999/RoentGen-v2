# RoentGen-v2: Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data

[![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/stanfordmimi/RoentGen-v2) [![arXiv](https://img.shields.io/badge/arXiv-2508.16783-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2508.16783) [![License](https://img.shields.io/github/license/stanfordmimi/RoentGen-v2?style=for-the-badge)](LICENSE) [<img src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/datasets-logo-light.svg" width="90" />](https://huggingface.co/datasets/stanfordmimi/RoentGen-v2-synthetic-dataset)

## 🧨Inference with diffusers

```python
import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = DiffusionPipeline.from_pretrained("stanfordmimi/RoentGen-v2")
pipe = pipe.to(device)

prompt = "50 year old female. Normal chest radiograph."
image = pipe(prompt).images[0]
```

## 🩻 Synthetic CXR Dataset
565k synthetic chest radiographs generated with RoentGen-v2 available on HuggingFace [here](https://huggingface.co/datasets/stanfordmimi/RoentGen-v2-synthetic-dataset).

![Visuals](assets/visual_examples.png)

## 🚀 Developer Mode
### Environment Setup
Strongly recommended to create a dedicated virtual environment for this project. 
A `requirements.txt` is provided. 
After you install the requirements via your package manager, it is important to run `pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 xformers --index-url https://download.pytorch.org/whl/cu126`. This downgrades torch in order to ensure version compatibility with `monai` and `torchxrayvision` packages. Tested and confirmed to work with wheels for `cu126`, `cu121` and `cu118`.

### Large-scale Inference
To run large-scale multi-gpu distributed inference, use the following commands.

Only inference, no quality check:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Only inference, no quality check (multi-gpu):
```bash
accelerate launch --num_processes=4 --multi-gpu --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Inference plus demographics quality check:
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference_w_quality_check.py \
 --config_file="./configs/infer_config_demo.yaml"
```

Inference plus demographics quality check (multi-gpu):
```bash
accelerate launch --num_processes=4 --multi-gpu --mixed_precision bf16 \
 roentgenv2/inference_code/run_inference_w_quality_check.py \
 --config_file="./configs/infer_config_demo.yaml"
```

### LoRA Finetuning on Your Own Images

This fork includes a lightweight LoRA fine-tuning script that uses
[`stanfordmimi/RoentGen-v2`](https://huggingface.co/stanfordmimi/RoentGen-v2)
as the base model and saves only LoRA adapter weights.

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you are using Colab, install a CUDA-compatible PyTorch build if needed:

```bash
pip install --upgrade torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu126
```

#### 2. Log in to Hugging Face

RoentGen-v2 is a gated model. Accept the model terms on Hugging Face first, then log in:

```bash
hf auth login
```

#### 3. Prepare the dataset

The LoRA script expects two flat folders: one for images and one for prompts. Each image must have a matching `.txt` file with the same basename.

```text
/content/temp_dataset_for_zip/images_512x512/
  case001.jpg
  case002.png

/content/temp_dataset_for_zip/reports/
  case001.txt
  case002.txt
```

Example prompt file:

```text
50 year old female. Normal chest radiograph.
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`.

#### 4. Check the config

The default LoRA config is:

```text
configs/train_lora_roentgen.yaml
```

It uses these default paths:

```yaml
image_dir: "/content/temp_dataset_for_zip/images_512x512"
prompt_dir: "/content/temp_dataset_for_zip/reports"
output_dir: "/content/drive/MyDrive/Projects/data/xray/train_01"
```

The default training settings are intended for limited-GPU Colab training:

```yaml
resolution: 512
train_batch_size: 1
gradient_accumulation_steps: 4
mixed_precision: fp16
gradient_checkpointing: true
unet_lora_rank: 8
unet_lora_alpha: 8
unet_learning_rate: 1.0e-4
train_text_encoder_lora: false
resume_from_checkpoint: "latest"
```

> **Recommended config.** For a stronger run, use
> `configs/train_lora_roentgen_recommended.yaml`. It trains longer
> (`max_train_steps: 6000`), uses a larger effective batch and LoRA rank,
> enables the validation curve, and uses the `center_crop` preprocessing (below).

To train both LoRA adapters, use
`configs/train_lora_roentgen_text_encoder.yaml`. It keeps separate capacity and
learning rates for the UNet and CLIP text encoder:

```yaml
unet_lora_rank: 16
text_encoder_lora_rank: 4
unet_learning_rate: 3.0e-5
text_encoder_learning_rate: 5.0e-6
train_text_encoder_lora: true
```

The lower text-encoder learning rate reduces the risk of damaging the pretrained
language representation. Joint training uses more VRAM than UNet-only LoRA;
keep `train_batch_size: 1` and gradient checkpointing enabled on a 16 GB GPU.

**Image preprocessing (`image_transform`).** Chest X-rays are non-square, so they
must be made square before training. Two modes are available:

```yaml
image_transform: center_crop   # recommended: resize + centered square crop, no borders
# image_transform: pad         # legacy: black-letterbox to square (adds black bars)
```

`center_crop` is preferred because the black bars added by `pad` are an
out-of-distribution artifact that inflates FID. Use `pad` only to reproduce
older runs.

#### 5. Start LoRA fine-tuning

```bash
accelerate launch roentgenv2/train_code/train_lora.py \
  --config_file configs/train_lora_roentgen.yaml
```

For the joint UNet + text-encoder run (`train_07`), use:

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  roentgenv2/train_code/train_lora.py \
  --config_file configs/train_lora_roentgen_text_encoder.yaml
```

Passing the Accelerate values explicitly avoids its "defaults used" warning.
The Pydantic `repr`/`frozen` messages emitted by some Colab dependency versions
are warnings and do not stop training.

The script saves checkpoints in:

```text
/content/drive/MyDrive/Projects/data/xray/train_01/checkpoint-*
```

If training stops, run the exact same command again. Because the config has:

```yaml
resume_from_checkpoint: "latest"
```

training will resume from the newest checkpoint in the same `output_dir`.

Final LoRA weights are saved to:

```text
/content/drive/MyDrive/Projects/data/xray/train_01/lora
```

For joint training, the single LoRA weights file in the `lora` directory
contains both UNet and text-encoder adapters. Each `checkpoint-*` directory also
contains both adapters, while `resume_from_checkpoint: "latest"` restores both
models and the optimizer/scheduler state.

#### 6. Use the fine-tuned LoRA for inference

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stanfordmimi/RoentGen-v2",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

pipe.load_lora_weights("/content/drive/MyDrive/Projects/data/xray/train_01/lora")

prompt = "50 year old female. Normal chest radiograph."
image = pipe(prompt).images[0]
```

#### 7. Monitor training (loss / validation curves)

Training writes a `metrics.json` into `output_dir` (updated at every checkpoint,
every validation, and at the end):

```json
{
  "meta":  { "unet_lora_rank": 16, "unet_learning_rate": 0.00003, "image_transform": "center_crop" },
  "train": { "step": [...], "loss": [...], "lr": [...], "text_encoder_lr": [...] },
  "val":   { "step": [...], "loss": [...] }
}
```

To get a **validation curve**, enable a held-out split in the config:

```yaml
val_split: 0.05          # hold out 5% of pairs for validation
validation_steps: 250    # compute + log validation loss every 250 steps
metrics_file: metrics.json
```

Validation loss uses fixed-seed noise, so it is comparable across steps (a real
"is it improving" signal rather than noise).

Turn `metrics.json` into a chart with:

```bash
python roentgenv2/train_code/plot_metrics.py \
  --metrics /content/drive/MyDrive/Projects/data/xray/train_recommended/metrics.json \
  --out training_curve.png --show-lr
```

The chart shows raw + EMA-smoothed training loss, validation-loss markers, and the
learning-rate schedule; the command also prints the best validation loss and the
step it occurred (useful for picking a checkpoint). Options: `--smooth 0.9` (EMA
factor, `0` = raw), `--show-lr` (overlay LR), `--dpi`.

### Full Finetuning Instructions

The original full UNet fine-tuning script is still available, but it needs more GPU memory than LoRA fine-tuning.
```bash
accelerate launch --num_processes=1 --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```

Finetuning (multi-gpu):
```bash
accelerate launch --num_processes 4 --multi_gpu --mixed_precision bf16 \
 roentgenv2/train_code/train.py \
 --config_file="./configs/train_config_demo.yaml"
```

### BioGPT text stage: IU X-Ray clinical report → RoentGen prompt

This project uses the proposal's intended order: create the text prompt first,
then give that generated prompt to RoentGen-v2 to create the synthetic image.
The text stage fine-tunes the official [`microsoft/biogpt`](https://huggingface.co/microsoft/biogpt)
checkpoint with the Claude-created `generated_prompt` as its target.

The supplied CSV contains the target prompt *and* the original report from which
Claude made it. Therefore this training task must be described accurately as
**clinical-report-to-standardized-prompt distillation**. It does not prove that
the model can create a report from only an image or a diagnosis. The default
input contains clinical sections (`indication`, `findings`, `impression`,
structured labels, and number of images), and deliberately excludes
`generated_prompt`, report/image identifiers, and cost/token metadata.

The existing folders shown below are the authority for the external split; the
script never re-randomizes their test examples:

```text
/content/xray-database/
  train/images/<id>.png
  train/labels/<id>.txt
  test/images/<id>.png
  test/labels/<id>.txt
```

The numeric names in your screenshots (for example `1002.png` and `1002.txt`)
are supported and are validated against CSV report IDs such as `CXR1002` before
anything is trained. A 10% validation set is sampled only from `train`; `test`
remains untouched until final evaluation.

#### 1. Install and place the two input folders/files

```bash
pip install -r requirements.txt

# In Colab, upload/copy the Claude CSV to this path (or change the command).
# /content/iu_xray_prompts_claude.csv
# Also make /content/xray-database/{train,test}/{images,labels} available.
```

`sacremoses` is included in `requirements.txt` because BioGPT's tokenizer
requires it. If the environment was installed before this update, run:

```bash
pip install sacremoses==0.1.1
```

#### 2. Prepare leakage-checked text splits

```bash
python roentgenv2/text_generation/prepare_dataset.py \
  --csv-file /content/iu_xray_prompts_claude.csv \
  --xray-database /content/xray-database \
  --output-dir /content/biogpt-data
```

Check `/content/biogpt-data/summary.json` before training. It records the exact
train/validation/test counts, the mapping audit, and whether the existing label
files exactly match the CSV's Claude targets. The CSV remains canonical; label
files are audited but are not silently substituted as targets.

If you deliberately want a stricter, less-leaky research variant based only on
diagnosis/context, replace the default inputs with:

```bash
--input-fields indication mesh_labels num_images
```

#### 3. Fine-tune BioGPT and save training metrics

Edit `configs/train_biogpt_iu_xray.yaml` only to set the persistent Google Drive
`output_dir` you want. Then run:

```bash
python roentgenv2/text_generation/train_biogpt.py \
  --config-file configs/train_biogpt_iu_xray.yaml
```

The default configuration is designed for one 16 GB T4 GPU: batch size 2,
gradient accumulation 8, FP16, and gradient checkpointing. It writes:

```text
<output_dir>/checkpoint-*/       resumable checkpoints
<output_dir>/metrics.json        loss/learning-rate history, validation, final held-out test metrics
<output_dir>/final_model/        final trained BioGPT model and tokenizer
<output_dir>/training_config.yaml
```

Re-run the same command after an interruption. `resume_from_checkpoint: latest`
finds the newest checkpoint automatically and continues updating the same
`metrics.json`.

#### 4. Generate test prompts for text-quality evaluation and RoentGen input

```bash
python roentgenv2/text_generation/generate_reports.py \
  --model-dir /content/drive/MyDrive/Projects/data/xray/biogpt_iu_xray_v1/final_model \
  --input-file /content/biogpt-data/test.jsonl \
  --output-csv /content/drive/MyDrive/Projects/data/xray/biogpt_iu_xray_v1/test_predictions.csv \
  --max-source-length 384
```

`test_predictions.csv` contains each held-out target and its BioGPT prediction.
Use the `prediction` column as the input prompt to RoentGen-v2. Keep this test
file separate from all training choices; it is what supports the proposal's
final text-quality and image-text-alignment evaluation.

### MedCLIP image-text alignment evaluation

After generating the held-out synthetic images, use the separate MedCLIP
evaluation environment. It deliberately runs outside the RoentGen/BioGPT
environment; the supplied requirements support current Python 3.12 Colab. See
[`docs/medclip-evaluation.md`](docs/medclip-evaluation.md) for the full command
and output definitions. The script writes per-pair matched and shuffled-negative
similarities, confidence intervals, AUROC, retrieval results, and a plot.

## 📎 Citation

If you find this repository useful for your work, please cite the following paper:

```bibtex
@misc{moroianu2025improvingperformancerobustnessfairness,
      title={Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data}, 
      author={Stefania L. Moroianu and Christian Bluethgen and Pierre Chambon and Mehdi Cherti and Jean-Benoit Delbrouck and Magdalini Paschali and Brandon Price and Judy Gichoya and Jenia Jitsev and Curtis P. Langlotz and Akshay S. Chaudhari},
      year={2025},
      eprint={2508.16783},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.16783}, 
}
```
