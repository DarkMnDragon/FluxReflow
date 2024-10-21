#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
import re
from contextlib import nullcontext
from pathlib import Path
from torch import Tensor

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

import subprocess
from huggingface_hub import login

def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).


## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = torch.autocast(accelerator.device.type)
    # autocast_ctx = nullcontext()

    with torch.inference_mode(), autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    # ReflowDataset
    parser.add_argument(
        "--reflow_data_root",
        type=str,
        default=None,
        required=True,
        help="The root directory of the reflow dataset.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_inference_steps",
        type=int,
        default=8,
        help="Number of inference steps to perform during validation.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=500,
        help=(
            "Run validation generation every X epochs, consisting of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--training_t_dist",
        type=str,
        default="u_shape",
        help="t distribution for training, choose be ['uniform', 'u_shape', 'logit_normal', 'flux_shift']",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-reflow-full-finetuing",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024*1024",
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=2, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy", "Adafactor"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="DarkMoonDragon/trained-flux",
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.use_dynamic_instance_reflow and (args.backward_reflow_threshold is not None or args.backward_update_steps is not None):
        raise ValueError("You must set --use_dynamic_instance_reflow if you want to use backward reflow.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class ReflowDataset(Dataset):
    """
    A dataset containing the reflow pairs and precomputed prompts.
    """

    def __init__(
        self,
        reflow_data_dir,
    ):
        self.img_root = Path(reflow_data_dir) / "img"
        self.prompt_root = Path(reflow_data_dir) / "prompt"
        self.prior_latent_root = Path(reflow_data_dir) / "z_0"
        self.img_latent_root = Path(reflow_data_dir) / "z_1"

        self.img_paths = sorted(self.img_root.glob('*.png'))
        self.prompt_paths = sorted(self.prompt_root.glob('*.pt'))
        self.prior_latent_paths = sorted(self.prior_latent_root.glob('*.pt'))
        self.img_latent_paths = sorted(self.img_latent_root.glob('*.pt'))

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        assert len(self.img_paths) == len(self.prompt_paths) == len(self.prior_latent_paths) == len(self.img_latent_paths), \
            "Number of images, prompts, z0 and z1 should be the same."

        print(f"Loaded {len(self)} reflow pairs.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Load image
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        image = exif_transpose(image)
        if not image.mode == "RGB": image = image.convert("RGB")
        image = self.img_transform(image)

        # Load prompt & embeddings
        prompt_path = self.prompt_paths[index]
        prompt_data = torch.load(prompt_path, map_location=torch.device("cpu"))
        prompt = prompt_data["prompt"]
        prompt_embed = prompt_data["prompt_embeds"].squeeze(0)
        pooled_prompt_embed = prompt_data["pooled_prompt_embeds"].squeeze(0)

        # Load reflow pairs
        prior_latent_path = self.prior_latent_paths[index]
        img_latent_path = self.img_latent_paths[index]
        prior_latent = torch.load(prior_latent_path, map_location=torch.device("cpu")).squeeze(0)
        img_latent = torch.load(img_latent_path, map_location=torch.device("cpu")).squeeze(0)

        return {
            "image": image,
            "prompt": prompt,
            "prompt_embeds": prompt_embed,
            "pooled_prompt_embeds": pooled_prompt_embed,
            "gaussian": prior_latent,  # prior latents in VAE space
            "latent": img_latent,      # img latents in VAE space
        }

def collate_fn(examples):
    batch = {}
    keys = examples[0].keys()

    for key in keys:
        values = [example[key] for example in examples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], str):
            batch[key] = values
        else:
            batch[key] = values
    
    return batch


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", variant=args.variant
    )

    # We only train the FluxTransformer2DModel
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)): # Maybe not to save T5EncoderModel
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        # unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        print("Skipping saving text_encoder_1, CLIPTextModelWithProjection")
                    else:
                        # unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                        print("Skipping saving text_encoder_2, T5EncoderModel")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw" or args.optimizer.lower() == "adafactor"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy, adafactor]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.optimizer.lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError:
            raise ImportError("To use Adafactor, please install the transformers library: `pip install transformers`")
        
        optimizer_class = Adafactor

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=0.01,
        )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    # Dataset and DataLoaders creation:
    train_dataset = ReflowDataset(
        reflow_data_dir=args.reflow_data_root,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    validation_prompt_hidden_states, validation_pooled_prompt_embeds, _ = compute_text_embeddings(
        args.validation_prompt, text_encoders, tokenizers
    )

    # Clear the memory here
    del tokenizers, text_encoders
    # Explicitly delete the objects as well, otherwise only the lists are deleted and the original references remain, preventing garbage collection
    del text_encoder_one, text_encoder_two
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-reflow-full-finetune"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def time_shift(mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def get_schedule(
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        timesteps = torch.linspace(1, 0, num_steps + 1)
        if shift:
            mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = time_shift(mu, 1.0, timesteps)
        return timesteps.tolist()

    def exponential_pdf(x, a):
        C = a / (torch.exp(torch.tensor(a)) - 1)
        return C * torch.exp(a * x)

    def u_shaped_t(num_samples, alpha=2.):
        alpha = torch.tensor(alpha, dtype=torch.float32)
        u = torch.rand(num_samples)
        t = -torch.log(1 - u * (1 - torch.exp(-alpha))) / alpha  # inverse cdf = torch.log(u * (torch.exp(torch.tensor(a)) - 1) / a) / a
        t = torch.cat([t, 1 - t], dim=0)
        t = t[torch.randperm(t.shape[0])]
        t = t[:num_samples]
        return t

    def sample_training_timesteps(t_dist, num_samples, image_seq_len):
        if t_dist == "uniform":
            timesteps = torch.rand(num_samples, dtype=torch.float32)
        elif t_dist == "u_shape":
            timesteps = u_shaped_t(num_samples)
        elif t_dist == "logit_normal":
            timesteps = torch.normal(mean=0., std=1., size=(num_samples,), dtype=torch.float32)
            timesteps = torch.nn.functional.sigmoid(timesteps)
        elif t_dist == "flux_shift":
            train_num_steps = torch.randint(1, 101, (1,)).item()
            t_list = get_schedule(
                            num_steps=train_num_steps,
                            image_seq_len=image_seq_len,
                            shift=True, # Uniform is in the above
                        )[:-1]
            timesteps = torch.tensor([t_list[torch.randint(0, len(t_list), (1,)).item()] for _ in range(latent.shape[0])])
        else:
            raise NotImplementedError(f"t_dist {t_dist} not implemented")
        return timesteps

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

        #     return {
        #     "image": image,
        #     "prompt": prompt,
        #     "prompt_embeds": prompt_embed,
        #     "pooled_prompt_embeds": pooled_prompt_embed,
        #     "gaussian": prior_latent,  # prior latents in VAE space
        #     "latent": img_latent,      # img latents in VAE space
        # }

            with accelerator.accumulate(models_to_accumulate):
                prompt = batch["prompt"]
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)
                latent = batch["latent"].to(dtype=weight_dtype)
                gaussian = batch["gaussian"].to(dtype=weight_dtype) 
                t_dist = args.training_t_dist
                print("train at t:", t, "t_dist:", t_dist, "prompt:", prompt)

                t = sample_training_timesteps(
                    t_dist, latent.shape[0], (latent.shape[2]//2) * (latent.shape[3]//2)
                    ).to(dtype=weight_dtype, device=latent.device)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                        latent.shape[0],
                        latent.shape[2],
                        latent.shape[3],
                        accelerator.device,
                        weight_dtype,
                    )

                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=prompt_embeds.device, dtype=weight_dtype)
                    
                noisy_latent = (1. - t[:, None, None, None]) * latent + t[:, None, None, None] * gaussian

                packed_noisy_latent_input = FluxPipeline._pack_latents(
                    noisy_latent,
                    batch_size=latent.shape[0],
                    num_channels_latents=latent.shape[1],
                    height=latent.shape[2],
                    width=latent.shape[3],
                )

                if transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(packed_noisy_latent_input.shape[0])
                else:
                    guidance = None

                # print("packed_noisy_latents_input", packed_noisy_latent_input.shape)
                # print("t5 encoder_hidden_states", prompt_embeds.shape)
                # print("clip pooled_projections", pooled_prompt_embeds.shape)
                # print("text_ids", text_ids.shape)
                # print("latent_image_ids", latent_image_ids.shape)

                model_pred = transformer(
                    hidden_states=packed_noisy_latent_input,
                    timestep=t, # No need to divide by 1000 here
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(noisy_latent.shape[2] * 16 / 2),
                    width=int(noisy_latent.shape[3] * 16 / 2),
                    vae_scale_factor=16,
                )

                # Rectified Flow loss
                target = gaussian - latent
                # Compute prior or instance loss
                loss = torch.mean((model_pred.float() - target.float()) ** 2)
                print(f"loss: {loss}")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                torch.cuda.empty_cache()
                gc.collect()
                pipeline = FluxPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=None,
                    text_encoder_2=None,
                    transformer=accelerator.unwrap_model(transformer),
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt_embeds": validation_prompt_hidden_states, 
                                 "pooled_prompt_embeds": validation_pooled_prompt_embeds,
                                 "num_inference_steps": args.num_validation_inference_steps}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                )
                torch.cuda.empty_cache()
                gc.collect()

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.to(torch.float32)
        
        pipeline = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, transformer=transformer)
        pipeline.save_pretrained(args.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.output_dir,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)