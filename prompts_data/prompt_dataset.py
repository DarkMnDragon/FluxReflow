import itertools
import random
import os
from torch.utils.data import Dataset
from pathlib import Path

import os
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, prompt_list=None, file_path=None):
        if prompt_list is not None:
            self.prompts = prompt_list
        elif file_path is not None:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f]
        else:
            raise ValueError("Either prompt_list or file_path must be provided.")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.prompts):
            raise IndexError(f"Index {idx} is out of range.")
        return {'prompt': self.prompts[idx]}
    
    def save_prompts_from_list(self, directory_path, file_name):
        """Save prompts as a txt file, one prompt per line"""
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)
    
        file_path = os.path.join(directory_path, file_name)
    
        with open(file_path, 'w', encoding='utf-8') as f:
            for prompt in self.prompts:
                f.write(f"{prompt}\n")
