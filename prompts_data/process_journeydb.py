import json
import random

def extract_prompts(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'prompt' in data:
                prompts.append(data['prompt'])
    return prompts

def sample_prompts(prompts, num_samples=5000):
    return random.sample(prompts, min(num_samples, len(prompts)))

def save_prompts_to_txt(prompts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")

file_path = 'your_jsonl_file_path.jsonl'
output_file = 'sampled_prompts.txt'

prompts = extract_prompts(file_path)

sampled_prompts = sample_prompts(prompts, 5000)

save_prompts_to_txt(sampled_prompts, output_file)

print(f"Sampled 5000 prompts have been saved to {output_file}")
