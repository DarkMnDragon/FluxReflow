import torch
import itertools
import random
import json
import os
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, prompt_list=None, file_path=None, key='prompts'):
        if prompt_list is not None:
            self.prompts = prompt_list
        elif file_path is not None:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.prompts = data[key]
        else:
            raise ValueError("Either prompt_list or file_path must be provided.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.prompts):
            raise IndexError(f"Index {idx} is out of range.")
        
        return {'prompt': self.prompts[idx]}

    def save_prompts_from_list(self, directory_path, file_name, key='prompts'):
        """Save the prompts to a file with the specified key."""
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)

        file_path = os.path.join(directory_path, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({key: self.prompts}, f, ensure_ascii=False, indent=4)  # Pretty format



def generate_dog_prompts():
    sizes = ["small", "medium-sized", "large", "tiny", "huge"]
    breeds = ["golden retriever", "pug", "beagle", "dalmatian", "bulldog", "chihuahua", "german shepherd", "husky"]
    actions = ["running", "sleeping", "barking", "playing", "jumping", "eating", "swimming"]
    locations = ["in a park", "on a beach", "in a garden", "on a couch", "in the backyard"]
    additional_details = ["with children playing", "under the sunshine", "next to a lake", "with a ball", "next to a cozy fire"]

    combinations = list(itertools.product(sizes, breeds, actions, locations, additional_details))
    prompt_templates = [f"A {size} {breed} dog {action} {location}, {detail}." for size, breed, action, location, detail in combinations]
    random.shuffle(prompt_templates)

    return prompt_templates

dataset = PromptDataset(file_path="/root/dreambooth_flux/aquacoltok.json")

print(dataset.__len__())