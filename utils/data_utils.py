from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torch

def deprocess(x):
    # [-1,1] -> [0, 1]
    return (x + 1.0) * 0.5

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x / 127.5) - 1.0

def augment(a_img, b_img):
    """
       Augment images - a is distorted
    """
    # randomly interpolate
    a = random.random()
    # flip image left right
    if random.random() < 0.25:
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    # flip image up down
    if random.random() < 0.25:
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img)
    return a_img, b_img

def getPaths(data_dir):
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d, filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)

def read_and_resize_pair(path_lr, path_hr, low_res=(240, 320), high_res=(240, 320)):
    img_lr = Image.open(path_lr).convert('RGB')
    img_lr = np.array(img_lr.resize(low_res, Image.BICUBIC))
    img_hr = Image.open(path_hr).convert('RGB')
    img_hr = np.array(img_hr.resize(high_res, Image.BICUBIC))
    return img_lr, img_hr

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
