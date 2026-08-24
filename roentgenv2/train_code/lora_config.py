from dataclasses import dataclass

import yaml


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
    unet_learning_rate: float = 1.0e-4
    text_encoder_learning_rate: float = 5.0e-6
    scale_lr: bool = False
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    max_train_steps: int = 1000
    num_train_epochs: int = 100
    max_train_samples: int = None

    unet_lora_rank: int = 8
    unet_lora_alpha: int = 8
    lora_dropout: float = 0.0
    text_encoder_lora_rank: int = 4
    text_encoder_lora_alpha: int = 4
    text_encoder_lora_dropout: float = 0.0
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

    image_transform: str = "center_crop"
    val_split: float = 0.0
    validation_steps: int = 0
    metrics_file: str = "metrics.json"

    logging_dir: str = "logs"
    report_to: str = "wandb"
    checkpointing_steps: int = 100
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: str = "latest"
    local_rank: int = -1

    def get_config(self):
        return self.__dict__


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    # Backward compatibility for configurations created before separate UNet
    # and text-encoder LoRA hyperparameters were supported.
    legacy_keys = {
        "learning_rate": "unet_learning_rate",
        "lora_rank": "unet_lora_rank",
        "lora_alpha": "unet_lora_alpha",
    }
    for legacy_key, replacement_key in legacy_keys.items():
        if legacy_key not in config:
            continue
        if replacement_key in config:
            raise ValueError(
                f"Config contains both {legacy_key!r} and {replacement_key!r}; "
                f"use only {replacement_key!r}."
            )
        config[replacement_key] = config.pop(legacy_key)

    return LoRATrainConfig(**config)
