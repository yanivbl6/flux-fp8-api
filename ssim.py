from pytorch_msssim import ssim

from torchvision import transforms

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
parser = argparse.ArgumentParser(description='Run parameters with their default values.')


## required, 
parser.add_argument('directory', type=str, help='a directory for processing')

parser.add_argument('-b','--baseline', type=str, help='a directory for baseline')
parser.add_argument('-r','--reference', type=str, help='a directory for baseline')



if __name__ == "__main__":
    # Load the pre-computed FID statistics for the COCO validation set

    args = parser.parse_args()    ## get 


    transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.baseline = os.path.join(args.directory, args.baseline)
    args.reference = os.path.join(args.directory, args.reference)

    assert os.path.exists(args.baseline), f"Baseline directory {args.baseline} does not exist"
    assert os.path.exists(args.reference), f"Reference directory {args.reference} does not exist"

    paths = [os.path.join(args.reference, f) for f in os.listdir(args.reference) if f.endswith(".jpg") or f.endswith(".png")]
    baseline_paths = [os.path.join(args.baseline, f) for f in os.listdir(args.reference) if f.endswith(".jpg") or f.endswith(".png")]



    N = len(paths)
    pbar = tqdm(enumerate(zip(paths, baseline_paths)), desc="Computing SSIM", total=len(paths))

    ssim_values = np.zeros(N)

    ssim_total = 0


    for i, (path, base_path) in pbar:
        img1 = Image.open(base_path)
        img2 = Image.open(path)

        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)

        ssim_value = ssim(img1, img2, data_range=1.0)

        ssim_values[i] = ssim_value.item()
        ssim_total += ssim_value.item()
        ssim_mean = ssim_total/(i+1)
        pbar.set_postfix({"SSIM": "%.02f" % ssim_mean})


    pbar = tqdm(enumerate(zip(paths, baseline_paths)), desc="Computing PSNR", total=len(paths))
    psnr_values = np.zeros(N)
    psnr_total = 0

    for i, (path, base_path) in pbar:
        img1 = Image.open(base_path)
        img2 = Image.open(path)


        img1 = transform(img1).unsqueeze(0).to(device)
        img2 = transform(img2).unsqueeze(0).to(device)

        mse = torch.mean((img1 - img2) ** 2)
        ##find maximum for every image in batch [b]
        max_pixel = img1.max()
        psnr = (20 * torch.log10(max_pixel / torch.sqrt(mse)))
        psnr_values[i] = psnr
        psnr_total += psnr
        if i % 10 == 0:
            psnr_mean = psnr_total/(i+1)
            pbar.set_postfix({"PSNR": "%.02f" % psnr_mean})


    

    print(f"Directory: {args.baseline}, Number of files: {len(baseline_paths)}")
    print(f"Directory: {args.reference}, Number of files: {len(paths)}")



    print(f"SSIM mean: {ssim_values.mean()}")
    print(f"SSIM std: {ssim_values.std()}")

    print(f"PSNR mean: {psnr_values.mean()}")
    print(f"PSNR std: {psnr_values.std()}")
