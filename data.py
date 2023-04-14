import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class PairedDataset(Dataset):
    def __init__(self, 
                 input_files, 
                 target_files, 
                 transform=transforms.Compose([
                     transforms.Resize((256, 256)),
                     transforms.PILToTensor(),
                     transforms.ConvertImageDtype(torch.float32),
                     transforms.Normalize((0, 0, 0), (1, 1, 1))
                 ])):
        self.input_files = input_files
        self.target_files = target_files
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        """Get the images"""
        input_img = Image.open(self.input_files[index]).convert('RGB')
        target_img = Image.open(self.target_files[index]).convert('RGB')

        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            torch.set_rng_state(state)

        return input_img, target_img
    
def get_dataset(split="train"):
    
    filenames = os.listdir(f"{split}/HE/")
    inputs = [f"{split}/HE/" + x for x in filenames]
    targets = [f"{split}/IHC/" + x for x in filenames]
    
    dataset = PairedDataset(inputs, targets)
    
    return dataset