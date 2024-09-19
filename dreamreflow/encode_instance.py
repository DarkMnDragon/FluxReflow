import torch
import argparse
import os
import json
from PIL import Image
from torchvision import transforms

from PIL import Image
from flux_utils import encode_imgs
from pathlib import Path

from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel
from diffusers import FluxPipeline, AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/root/autodl-tmp/FLUX-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/data/dog-instance",
        help="Path to save the generated reflow pairs.",
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
    if args.generation_precision == "bf16": weight_dtype = torch.bfloat16
    elif args.generation_precision == "fp32": weight_dtype = torch.float
    else: weight_dtype = torch.float16
    print(f"Using {weight_dtype} for model weights.")

    output_dir = Path(args.output_dir)

    img_dir = output_dir / 'img'
    prompt_dir = output_dir / 'prompt'
    z1_dir = output_dir / 'z1'
    template_json_path = output_dir / 'instance_prompts.json'

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(z1_dir, exist_ok=True)
    
    if not os.path.exists(template_json_path):
        print(f"Cannot find '{template_json_path}', please run preprocess_instance.py first.")
        return
    
    with open(template_json_path, 'r', encoding='utf-8') as json_file:
        prompts = json.load(json_file)

    image_files = sorted(prompts.keys())

    empty_prompts = [fname for fname, prompt in prompts.items() if not prompt.strip()]
    if empty_prompts:
        print("The following images have not been filled in with prompts, please fill in the prompts in prompts.json:")
        for fname in empty_prompts:
            print(f"- {fname}")
        return
    
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        torch_dtype=weight_dtype,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        torch_dtype=weight_dtype,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    pipeline = FluxPipeline(
        scheduler=None,
        tokenizer=tokenizer_one,
        text_encoder=text_encoder_one,
        tokenizer_2=tokenizer_two,
        text_encoder_2=text_encoder_two,
        vae=vae,
        transformer=None,
    )
    pipeline.to(weight_dtype).to("cuda")
    print("Successfully loaded pipeline.")
    
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize([0.5], [0.5]),  
    ])

    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(img_dir, image_file)
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)  # [C, H, W]
        img_tensor = img_tensor.unsqueeze(0).to("cuda").to(weight_dtype)  # [1, C, H, W]

        with torch.no_grad():
            img_latent = encode_imgs(img_tensor, pipeline.vae)  # [1, 16, H//8, W//8]
            img_latent = img_latent.detach().cpu()

        z1_path = z1_dir / f'z_1_{idx+1:05d}.pt'
        torch.save(img_latent, z1_path)

        prompt = prompts[image_file]

        print(f"Processing {image_file} with prompt: {prompt}")

        with torch.no_grad():
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

            torch.save({
                "prompt": prompt,
                "prompt_embeds": prompt_embeds.detach().clone().cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.detach().clone().cpu(),
            }, prompt_dir / f'prompt_{idx+1:05d}.pt')

        print(f"Processed {image_file}, saved as {z1_path} and prompt_{idx+1:05d}.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)