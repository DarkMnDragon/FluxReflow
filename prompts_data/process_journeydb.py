import json
import random

# https://journeydb.github.io

def extract_captions(file_path):
    captions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line) # NOTE: Use caption instead of original prompt
            if 'Task2' in data and 'Caption' in data['Task2']:
                captions.append(data['Task2']['Caption'])
    return captions

def filter_prompts_by_length(prompts, min_words=10):
    """filter out prompts with less than or equal to the specified number of words."""
    return [prompt for prompt in prompts if len(prompt.split()) > min_words]

def sample_prompts(prompts, num_samples=5000):
    return random.sample(prompts, min(num_samples, len(prompts)))

def save_prompts_to_txt(prompts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")

file_path = '/Users/darkmoonbook/Downloads/train_anno_realease_repath.jsonl'
output_file = 'sampled_prompts.txt'

prompts = extract_captions(file_path)

prompts = filter_prompts_by_length(prompts, min_words=10)

sampled_prompts = sample_prompts(prompts, 5000)

save_prompts_to_txt(sampled_prompts, output_file)

print(f"Sampled {len(sampled_prompts)} prompts to {output_file}")