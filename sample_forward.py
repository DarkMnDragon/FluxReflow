import torch
import argparse
import math
import itertools
import random

from torch import Tensor
from typing import Callable
from PIL import Image

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, CLIPTextModel, T5EncoderModel

torch.backends.cuda.matmul.allow_tf32 = True

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="scheduler",
)
tokenizer_one = CLIPTokenizer.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="tokenizer",
)
tokenizer_two = T5TokenizerFast.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="tokenizer_2",
)
text_encoder_one = CLIPTextModel.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="text_encoder",
)
text_encoder_two = T5EncoderModel.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="text_encoder_2",
)
vae = AutoencoderKL.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="vae",
)
transformer = FluxTransformer2DModel.from_pretrained(
    "/root/autodl-tmp/data/FLUX-dev",
    subfolder="transformer",
)

transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

sizes = ["small", "medium-sized", "large", "tiny", "huge"]
breeds = ["golden retriever", "pug", "beagle", "dalmatian", "bulldog", "chihuahua", "german shepherd", "husky"]
actions = ["running", "sleeping", "barking", "playing", "jumping", "eating", "swimming"]
locations = ["in a park", "on a beach", "in a garden", "on a couch", "in the backyard"]
additional_details = ["with children playing", "under the sunshine", "next to a lake", "with a ball", "next to a cozy fire"]

combinations = list(itertools.product(sizes, breeds, actions, locations, additional_details))
prompt_templates = [f"A {size} {breed} dog {action} {location}, {detail}." for size, breed, action, location, detail in combinations]
random.shuffle(prompt_templates)


@torch.inference_mode()
def encode_imgs(imgs, weight_dtype=torch.bfloat16):
    latents = vae.encode(imgs).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    latents = latents.to(dtype=weight_dtype)
    return latents

@torch.inference_mode()
def decode_imgs(latents, pipeline):
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        metavar="N",
        help="number of samples to generate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="resolution of the generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="guidance scale",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

@torch.inference_mode()
def main(args):
    print(prompt_templates)
    weight_dtype = torch.bfloat16

    pipeline = FluxPipeline(
        scheduler=scheduler,
        tokenizer=tokenizer_one,
        text_encoder=text_encoder_one,
        tokenizer_2=tokenizer_two,
        text_encoder_2=text_encoder_two,
        vae=vae,
        transformer=transformer,
    ).to(weight_dtype).to("cuda")

    for step in range(args.num_samples):
        print(f"Generating sample {step + 1}/{args.num_samples}...")
        (   prompt_embeds, # save
            pooled_prompt_embeds, # save
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompt_templates[step % len(prompt_templates)],
            prompt_2=prompt_templates[step % len(prompt_templates)],
            device=pipeline.device,
            max_sequence_length=512,
        )

        torch.save({
            "prompt": prompt_templates[step % len(prompt_templates)],
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }, f"/root/autodl-tmp/reflow/prompt/prompt_{step}.pt")

        noise = get_noise( # save, shape [num_samples, 16, resolution // 8, resolution // 8]
            num_samples=1,
            height=args.resolution,
            width=args.resolution,
            device="cuda",
            dtype=weight_dtype,
            seed=step,
        )

        torch.save(noise, f"/root/autodl-tmp/reflow/z_0/z_0_{step}.pt")

        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            noise.shape[0],
            noise.shape[2],
            noise.shape[3],
            noise.device,
            weight_dtype,
        )

        packed_latents = FluxPipeline._pack_latents( # shape [num_samples, (resolution // 16 * resolution // 16), 16 * 2 * 2]
            noise,
            batch_size=noise.shape[0],
            num_channels_latents=noise.shape[1],
            height=noise.shape[2],
            width=noise.shape[3],
        )

        timesteps = timesteps = get_schedule( # shape: [num_inference_steps]
            num_steps=args.num_inference_steps,
            image_seq_len=(args.resolution // 16) * (args.resolution // 16), # vae // 8 and patchify // 2
            shift=True,  # Set True for Flux-dev, False for Flux-schnell
        )

        with pipeline.progress_bar(total=args.num_inference_steps) as progress_bar:
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)
                guidance_vec = torch.full((packed_latents.shape[0],), args.guidance_scale, device=packed_latents.device, dtype=packed_latents.dtype)
                pred = transformer(
                    hidden_states=packed_latents, 
                    timestep=t_vec,        
                    guidance=guidance_vec, 
                    pooled_projections=pooled_prompt_embeds, 
                    encoder_hidden_states=prompt_embeds,     
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=pipeline,
                )[0]
                packed_latents = packed_latents + (t_prev - t_curr) * pred
                # imgs_pred = latents - t_curr * pred
                # imgs_pred_list.append(imgs_pred)
                progress_bar.update()

        assert noise.shape[2]*8 == args.resolution and noise.shape[3]*8 == args.resolution
        assert pipeline.vae_scale_factor == 16
        img_latents = FluxPipeline._unpack_latents( # save, shape [num_samples, 16, resolution//8, resolution//8]
            packed_latents,
            height=args.resolution,
            width=args.resolution,
            vae_scale_factor=pipeline.vae_scale_factor,
        )

        torch.save(img_latents, f"/root/autodl-tmp/reflow/z_1/z_1_{step}.pt")

        imgs = decode_imgs(img_latents, pipeline)[0]
        imgs.save(f"/root/autodl-tmp/reflow/imgs/sample_{step}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)