import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class uwcc(Dataset):
    def __init__(self, ori_files, ucc_files, transform=None):
        self.ori_files = ori_files
        self.ucc_files = ucc_files
        self.transform = transform

        self.ori_images = []
        self.ucc_images = []

        # Load image file paths
        for ori_file, ucc_file in zip(self.ori_files, self.ucc_files):
            self.ori_images.append(ori_file)
            self.ucc_images.append(ucc_file)

        # Sort to ensure corresponding pairs
        self.ori_images.sort()
        self.ucc_images.sort()

        # Check if the lengths of the image lists are equal
        if len(self.ori_images) != len(self.ucc_images):
            raise ValueError("The number of original images and ground truth images do not match.")

    def __len__(self):
        return len(self.ori_images)

    def __getitem__(self, idx):
        ori_image_path = self.ori_images[idx]
        ucc_image_path = self.ucc_images[idx]
        
        try:
            ori_image = Image.open(ori_image_path).convert('RGB')
            ucc_image = Image.open(ucc_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image pair: {ori_image_path}, {ucc_image_path}")
            print(e)
            return None, None

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