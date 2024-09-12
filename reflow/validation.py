import torch
import argparse
import random
import numpy as np

from PIL import Image
from flux_utils import get_models, get_noise, get_schedule, decode_imgs
from prompt_dataset import PromptDataset
from diffusers import FluxPipeline
from torch.utils.data import DataLoader
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        default="/root/autodl-tmp/data/3rf-scratch.safetensors",
        help="Path to LoRA ckpt",
    )
    parser.add_argument(
        "--output_img_dir",
        type=str,
        default="/root/autodl-tmp/samples/3rf-scratch",
        help="Path to save the generated images.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/root/dreambooth_flux/prompts_data/gpt4o_various_prompts.json",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="number of epochs",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="resolution of the generated images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="resolution of the generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="guidance scale",
    )
    parser.add_argument(
        "--generation_precision",
        type=str,
        default="bf16",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to fp16 if a GPU is available else fp32."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


@torch.inference_mode()
def main(args):
    if args.generation_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.generation_precision == "fp32":
        weight_dtype = torch.float
    else:
        weight_dtype = torch.float16
    print(f"Using {weight_dtype} for generation")

    dataset = PromptDataset(file_path=args.prompt_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    (
        scheduler,
        tokenizer_one,
        tokenizer_two,
        text_encoder_one,
        text_encoder_two,
        vae,
        transformer
    ) = get_models(args.pretrained_model_name_or_path, weight_dtype)

    pipeline = FluxPipeline(
        scheduler=scheduler,
        tokenizer=tokenizer_one,
        text_encoder=text_encoder_one,
        tokenizer_2=tokenizer_two,
        text_encoder_2=text_encoder_two,
        vae=vae,
        transformer=transformer,
    )
    if args.lora_name_or_path is not None:
        print("Loading LoRA weights from", args.lora_name_or_path)
        pipeline.load_lora_weights(args.lora_name_or_path)
    else:
        print("No LoRA weights loaded")
    pipeline.to(weight_dtype).to("cuda")

    step = 0

    output_img_dir = Path(args.output_img_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch in dataloader:
            step += 1
            print(f"Generating sample {step} / {args.num_epochs * dataset.__len__()}...")
            print(f"Prompt: {batch['prompt']}")

            prompt = batch["prompt"]
            (
                prompt_embeds,         # save
                pooled_prompt_embeds,  # save
                text_ids,
            ) = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=pipeline.device,
                max_sequence_length=512,
            )

            noise = get_noise(  # save, shape [num_samples, 16, height // 8, width // 8]
                num_samples=1,
                height=args.height,
                width=args.width,
                device="cuda",
                dtype=weight_dtype,
                seed=step,
            )

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                noise.shape[0],
                noise.shape[2],
                noise.shape[3],
                noise.device,
                weight_dtype,
            )

            packed_latents = FluxPipeline._pack_latents(
                # [num_samples, (height // 16 * width // 16), 16 * 2 * 2]
                noise,
                batch_size=noise.shape[0],
                num_channels_latents=noise.shape[1],
                height=noise.shape[2],
                width=noise.shape[3],
            )

            timesteps = timesteps = get_schedule(  # shape: [num_inference_steps]
                num_steps=args.num_inference_steps,
                image_seq_len=(args.height // 16) * (args.width // 16),  # vae // 8 and patchify // 2
                shift=True,  # Set True for Flux-dev, False for Flux-schnell
            )

            with pipeline.progress_bar(total=args.num_inference_steps) as progress_bar:
                for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                    t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype,
                                       device=packed_latents.device)
                    guidance_vec = torch.full((packed_latents.shape[0],), args.guidance_scale,
                                              device=packed_latents.device,
                                              dtype=packed_latents.dtype)
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
                    progress_bar.update()

            assert noise.shape[2] * 8 == args.height and noise.shape[3] * 8 == args.width
            assert pipeline.vae_scale_factor == 16
            img_latents = FluxPipeline._unpack_latents(  # save, shape [num_samples, 16, height//8, width//8]
                packed_latents,
                height=args.height,
                width=args.width,
                vae_scale_factor=pipeline.vae_scale_factor,
            )

            imgs = decode_imgs(img_latents, vae, pipeline)[0]
            imgs.save(output_img_dir / f"img_{step:04d}.png")


if __name__ == "__main__":
    set_seed(0)
    args = parse_args()
    main(args)
