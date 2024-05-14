import os
import ntpath
import numpy as np
import imageio.v2 as imageio
from utils.data_utils import getPaths
from utils.uqim_utils import getUIQM
from utils.ssm_psnr_utils import getSSIM, getPSNR
from PIL import Image

# Data paths
GTr_im_dir = '/content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data/train/reference'  # Ground truth image directory
GEN_im_dir = '/content/drive/MyDrive/Underwater/UnderwaterIEAndSR/data/Output'  # Generated image directory

# Get paths for generated images
GEN_paths = getPaths(GEN_im_dir)

# Ensure filenames in both directories match
def match_filenames(gt_dir, gen_dir):
    gt_files = {ntpath.basename(path).split('.')[0]: path for path in getPaths(gt_dir)}
    gen_files = {ntpath.basename(path).split('.')[0]: path for path in getPaths(gen_dir)}  # No suffix
    common_files = set(gt_files.keys()).intersection(set(gen_files.keys()))
    print(f"Common files: {len(common_files)} found")  # Debug statement
    return [(gt_files[f], gen_files[f]) for f in common_files]

# Resize images to the same dimensions
def resize_image(image, size=(240, 320)):
    return np.array(Image.fromarray(image).resize(size, Image.BICUBIC))

# Measures UQIM for all images in a directory
def measure_UIQMs(dir_name):
    paths = getPaths(dir_name)
    uqims = []
    for img_path in paths:
        im = imageio.imread(img_path)
        if im is not None and im.size > 0:
            uqims.append(getUIQM(im))
        else:
            print(f"Warning: Failed to read image {img_path}")
    return np.array(uqims)

# Compares average SSIM and PSNR
def measure_SSIM_PSNRs(GT_dir, Gen_dir):
    file_pairs = match_filenames(GT_dir, Gen_dir)
    ssims, psnrs = [], []
    for gt_path, gen_path in file_pairs:
        print(f"Processing pair: GT: {gt_path}, GEN: {gen_path}")  # Debug statement
        r_im = imageio.imread(gt_path)
        g_im = imageio.imread(gen_path)
        if r_im is not None and g_im is not None and r_im.size > 0 and g_im.size > 0:
            print(f"Original GT Image shape: {r_im.shape}, Generated Image shape: {g_im.shape}")  # Debug statement
            r_im = resize_image(r_im)
            g_im = resize_image(g_im)
            print(f"Resized GT Image shape: {r_im.shape}, Resized Generated Image shape: {g_im.shape}")  # Debug statement
            assert r_im.shape == g_im.shape, "The images should be of the same size"
            ssim = getSSIM(r_im, g_im)
            psnr = getPSNR(r_im, g_im)
            ssims.append(ssim)
            psnrs.append(psnr)
        else:
            print(f"Warning: Failed to read image pair {gt_path}, {gen_path}")
    return np.array(ssims), np.array(psnrs)

# Compute SSIM and PSNR
SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(GTr_im_dir, GEN_im_dir)
if SSIM_measures.size > 0:
    print("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
else:
    print("SSIM measurements could not be computed.")

if PSNR_measures.size > 0:
    print("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
else:
    print("PSNR measurements could not be computed.")

# Compute and compare UIQMs
g_truth = measure_UIQMs(GTr_im_dir)
if g_truth.size > 0:
    print("G. Truth UQIM  >> Mean: {0} std: {1}".format(np.mean(g_truth), np.std(g_truth)))
else:
    print("Ground truth UQIM measurements could not be computed.")

gen_uqims = measure_UIQMs(GEN_im_dir)
if gen_uqims.size > 0:
    print("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
else:
    print("Generated UQIM measurements could not be computed.")
