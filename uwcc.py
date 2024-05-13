import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class uwcc(Dataset):
    def __init__(self, ori_dirs, ucc_dirs, train=True, transform=None):
        self.ori_dirs = ori_dirs
        self.ucc_dirs = ucc_dirs
        self.train = train
        self.transform = transform

        self.ori_images = []
        self.ucc_images = []
        
        # Load image file paths
        for ori_dir, ucc_dir in zip(ori_dirs, ucc_dirs):
            ori_files = [os.path.join(ori_dir, f) for f in os.listdir(ori_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            ucc_files = [os.path.join(ucc_dir, f) for f in os.listdir(ucc_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            self.ori_images.extend(ori_files)
            self.ucc_images.extend(ucc_files)

        # Sort to ensure corresponding pairs
        self.ori_images.sort()
        self.ucc_images.sort()

    def __len__(self):
        return len(self.ori_images)

    def __getitem__(self, idx):
        ori_image_path = self.ori_images[idx]
        ucc_image_path = self.ucc_images[idx]
        
        ori_image = Image.open(ori_image_path).convert('RGB')
        ucc_image = Image.open(ucc_image_path).convert('RGB')

        if self.transform:
            ori_image = self.transform(ori_image)
            ucc_image = self.transform(ucc_image)
        else:
            ori_image = self.default_transform(ori_image)
            ucc_image = self.default_transform(ucc_image)

        return ori_image, ucc_image

    def default_transform(self, image):
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])
        return transform(image)
