import random

# https://dagshub.com/datasets/laion-aesthetics-v2-6-5/

def extract_prompts(tsv_file_path):
    prompts = []
    with open(tsv_file_path, 'r', encoding='utf-8') as tsv_file:
        for line in tsv_file:
            line = line.strip()
            if line:
                columns = line.split('\t')
                if len(columns) >= 2:
                    caption = columns[1]
                    prompts.append(caption)
    return prompts

def filter_prompts_by_length(prompts, min_words=10):
    """filter out prompts with less than or equal to the specified number of words."""
    return [prompt for prompt in prompts if len(prompt.split()) > min_words]

def sample_prompts(prompts, num_samples=5000):
    return random.sample(prompts, min(num_samples, len(prompts)))

def save_prompts_to_txt(prompts, output_file):
    with open(output_file, 'w', encoding='utf-8') as txt_file:
        for caption in prompts:
            txt_file.write(f"{caption}\n")
    print(f"{len(prompts)} 个 prompts 已保存到 {output_file}")

tsv_file_path = '/Users/darkmoonbook/Downloads/labels.tsv'
output_file = 'sampled_prompts.txt'

prompts = extract_prompts(tsv_file_path)

prompts = filter_prompts_by_length(prompts, min_words=10)

sampled_prompts = sample_prompts(prompts, 5000)

save_prompts_to_txt(sampled_prompts, output_file)
