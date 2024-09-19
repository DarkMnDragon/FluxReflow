import torch
import math

from torch import Tensor
from typing import Callable

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, CLIPTextModel, T5EncoderModel

@torch.inference_mode()
def encode_imgs(imgs, vae, weight_dtype=torch.bfloat16):  # img, shape [B, C, H, W]
    latents = vae.encode(imgs).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = latents.to(dtype=weight_dtype)
    return latents


@torch.inference_mode()
def decode_imgs(latents, vae, pipeline):  # img latents, shape [B, 16, H//8, W//8]
    imgs = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    imgs = vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs


def get_noise(
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
):
    return torch.randn(  # [B, 16, H // 8, W // 8], latents after VAE
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
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
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)
    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def get_models(pretrained_model_name_or_path, dtype):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=dtype,
    )
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        torch_dtype=dtype,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        torch_dtype=dtype,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    return (
        scheduler,
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        vae,
        transformer
    )
