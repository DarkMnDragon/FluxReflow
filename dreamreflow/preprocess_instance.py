import os
from PIL import Image
import json
import argparse
from pathlib import Path

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--instance_dir",
        type=str,
        default="/root/dreambooth_flux/dreamreflow/dog",
        help="Path to the directory containing the instance images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/data/dog-instance",
        help="Path to save the processed images.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    instance_dir = Path(args.instance_dir)
    img_dir = Path(args.output_dir) / 'img'
    template_json_path = Path(args.output_dir) / 'instance_prompts.json'
    resolution = 1024

    os.makedirs(img_dir, exist_ok=True)

    image_files = [f for f in os.listdir(instance_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    prompts_template = {}

    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(instance_dir, image_file)
        img = Image.open(img_path).convert('RGB')

        img_processed = resize_and_center_crop(img, (resolution, resolution))

        new_filename = f'img_{idx+1:05d}.png'
        new_img_path = os.path.join(img_dir, new_filename)
        img_processed.save(new_img_path)

        prompts_template[new_filename] = ""
        
    with open(template_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(prompts_template, json_file, indent=4, ensure_ascii=False)

    print(f"Processed {len(image_files)} images and saved in '{img_dir}' directory.")
    print(f"Created template JSON file '{template_json_path}', please fill in the corresponding prompts.")

def resize_and_center_crop(img, target_size):
    original_width, original_height = img.size
    target_width, target_height = target_size

    ratio = max(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))

    img_resized = img.resize(new_size, Image.LANCZOS)

    left = (img_resized.width - target_width) / 2
    top = (img_resized.height - target_height) / 2
    right = left + target_width
    bottom = top + target_height

    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped

if __name__ == "__main__":
    args = parse_args()
    main(args)
