import argparse
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from tqdm.auto import tqdm

check_min_version("0.35.0")

if is_wandb_available():
    import wandb  # noqa: F401

logger = get_logger(__name__, log_level="INFO")


@dataclass
class LoRATrainConfig:
    pretrained_model_name_or_path: str = "stanfordmimi/RoentGen-v2"
    revision: str = None
    variant: str = None
    use_auth_token: str = None
    cache_dir: str = None

    image_dir: str = "/content/temp_dataset_for_zip/images_512x512"
    prompt_dir: str = "/content/temp_dataset_for_zip/reports"
    output_dir: str = "/content/drive/MyDrive/Projects/data/xray/train_01"

    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    learning_rate: float = 1.0e-4
    scale_lr: bool = False
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    max_train_steps: int = 1000
    num_train_epochs: int = 100
    max_train_samples: int = None

    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    train_text_encoder_lora: bool = False

    seed: int = 873
    dataloader_num_workers: int = 0
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1.0e-2
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    allow_tf32: bool = False
    enable_xformers_memory_efficient_attention: bool = True
    prediction_type: str = None
    noise_offset: float = 0.0

    logging_dir: str = "logs"
    report_to: str = "wandb"
    checkpointing_steps: int = 100
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: str = "latest"
    local_rank: int = -1

    def get_config(self):
        return self.__dict__


class SquarePad:
    def __call__(self, image):
        _, height, width = image.shape
        max_wh = max(width, height)
        pad_left = (max_wh - width) // 2
        pad_right = max_wh - width - pad_left
        pad_top = (max_wh - height) // 2
        pad_bottom = max_wh - height - pad_top
        return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)


class ImagePromptDirectoryDataset(Dataset):
    image_extensions = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_dir, prompt_dir, tokenizer, resolution, max_train_samples=None):
        self.image_dir = Path(image_dir)
        self.prompt_dir = Path(prompt_dir)
        self.tokenizer = tokenizer

        if not self.image_dir.exists():
            raise FileNotFoundError(f"image_dir does not exist: {self.image_dir}")
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"prompt_dir does not exist: {self.prompt_dir}")

        image_files = sorted(
            p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in self.image_extensions
        )

        self.samples = []
        for image_path in image_files:
            prompt_path = self.prompt_dir / f"{image_path.stem}.txt"
            if prompt_path.exists():
                self.samples.append((image_path, prompt_path))
            else:
                print(f"Warning: no matching prompt for image {image_path.name}; skipping.")

        if max_train_samples is not None:
            self.samples = self.samples[:max_train_samples]

        if not self.samples:
            raise ValueError(
                "No valid image-text pairs found. Images must be .jpg, .jpeg, or .png, "
                "and each image basename must have a matching .txt prompt file."
            )

        self.image_transforms = Compose(
            [
                ToTensor(),
                SquarePad(),
                Resize(resolution, interpolation=InterpolationMode.BILINEAR),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, prompt_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transforms(image)

        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            prompt = prompt_file.read().strip()

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
        }


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return LoRATrainConfig(**config)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for RoentGen-v2.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    config = load_config(args.config_file)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.local_rank:
        config.local_rank = env_local_rank

    return config


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return checkpoints[-1] if checkpoints else None


def prune_old_checkpoints(output_dir, checkpoints_total_limit):
    if checkpoints_total_limit is None:
        return

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) < checkpoints_total_limit:
        return

    num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    for checkpoint in checkpoints[:num_to_remove]:
        shutil.rmtree(os.path.join(output_dir, checkpoint))


def save_lora_weights(accelerator, unet, save_directory):
    unwrapped_unet = unwrap_model(accelerator, unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=save_directory,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )


def main(args):
    if args.train_text_encoder_lora:
        raise NotImplementedError("Text-encoder LoRA is not implemented yet. Use train_text_encoder_lora: false.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config_lora.yaml"), "w", encoding="utf-8") as config_file:
            yaml.dump(args.get_config(), config_file)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    kwargs_from_pretrained = {}
    if args.cache_dir is not None:
        kwargs_from_pretrained["cache_dir"] = args.cache_dir
    if args.revision is not None:
        kwargs_from_pretrained["revision"] = args.revision
    if args.variant is not None:
        kwargs_from_pretrained["variant"] = args.variant
    if args.use_auth_token is not None:
        kwargs_from_pretrained["token"] = args.use_auth_token

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        **kwargs_from_pretrained,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        **kwargs_from_pretrained,
    )

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory-efficient attention.")
        else:
            logger.info("xFormers requested but not available; continuing without it.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    lora_layers = [p for p in unet.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: pip install bitsandbytes") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = ImagePromptDirectoryDataset(
        image_dir=args.image_dir,
        prompt_dir=args.prompt_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
        max_train_samples=args.max_train_samples,
    )
    if accelerator.is_main_process:
        logger.info(f"Matched image/prompt pairs: {len(train_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("roentgen-v2-lora", config=args.get_config())

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            checkpoint_path = get_latest_checkpoint(args.output_dir)
        else:
            checkpoint_path = os.path.basename(args.resume_from_checkpoint)

        if checkpoint_path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new run.")
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            accelerator.load_state(os.path.join(args.output_dir, checkpoint_path))
            global_step = int(checkpoint_path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running LoRA training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        prune_old_checkpoints(args.output_dir, args.checkpoints_total_limit)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_lora_weights(accelerator, unet, save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_lora_dir = os.path.join(args.output_dir, "lora")
        save_lora_weights(accelerator, unet, final_lora_dir)
        logger.info(f"Saved final LoRA weights to {final_lora_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
