from pytorch_msssim import ssim

from torchvision import transforms

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


parser = argparse.ArgumentParser(description='Run parameters with their default values.')


## required, 
parser.add_argument('directory', type=str, help='a directory for processing')
parser.add_argument('-n','--images', type=int,  help='maximum number of images', default=7)
parser.add_argument('-s','--start', type=int,  help='starting index', default=0)

parser.add_argument('--rescaled', action='store_true', help='rescaled images')


if __name__ == "__main__":
    # Load the pre-computed FID statistics for the COCO validation set

    args = parser.parse_args()    ## get 


    transform = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    assert os.path.exists(args.directory), f"Directory {args.directory} does not exist"
    subdirs = os.listdir(args.directory)

    assert len(subdirs) > 0, f"Directory {args.directory} is empty"
    

    subdirs=["sdx50_config-bl","sdx50_config-dev", "sdx50_config-dev-sr","sdx50_config-dev-wsr", "sdx50_config-dev-unbiased"]
    names=["Baseline", "M3E4", "SR", "WSR", "SR+WSR"]
    psnr_scores = ["", "23.66" , "23.99", "23.51",  "24.11" ]

    if args.rescaled:
        subdirs = [f"{subdir}-rescaled" if i>0 else subdir for i,subdir in enumerate(subdirs) ]
        names=["Baseline", "M3E4 (s)", "SR (s)", "WSR (s)", "SR+WSR (s)"]
        psnr_scores = ["", "23.97" , "23.90", "23.82",  "24.15" ]

    ##subdirs[-1] = "20x5_sd_plms_50_wsr4__"
    ## ssim_scores = [1.00, 0.605, 0.729, 0.749,  0.753 ]


    scores = psnr_scores
    original = None

    all_paths = []
    cols = 0
    active_subdirs = []
    # for subdir in subdirs:
    #     subdir = os.path.join(args.directory, subdir)
    #     subdir = os.path.join(subdir, "samples")

    #     if not os.path.exists(subdir) or len(os.listdir(subdir)) == 0:
    #         continue

    #     if original is None:
    #         original = subdir
    #         files = [f for f in os.listdir(subdir) if f.endswith(".jpg") or f.endswith(".png")]

    #     ## remove files if not os.path.exists(os.path.join(args.reference, f)) 
    #     files = [f for f in files if os.path.exists(os.path.join(subdir, f))]
    #     active_subdirs.append(subdir)
    #     cols = cols + 1

    # if len(files) > args.images+args.start:
    #     files = files[args.start:args.images+args.start]
    # else:
    #     files = files[-args.start:]
    active_subdirs = [os.path.join(args.directory, subdir) for subdir in subdirs]
    cols = len(active_subdirs)

    files = ["image_%d.png" % i for i in range(args.start, args.start+args.images)]



    ## need to make grid of all images, col (direcory) x row (files)
    
    save_dir = "grids"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    if args.rescaled:
        c = canvas.Canvas(f"{save_dir}/grid-{args.directory}-{args.start}-{args.start+args.images}-rescaled.pdf", pagesize=letter)
    else:    
        c = canvas.Canvas(f"{save_dir}/grid-{args.directory}-{args.start}-{args.start+args.images}.pdf", pagesize=letter)
    width, height = letter
    margin = 10

    image_size_x = (width - 2 * margin) / cols
    image_size_y = (height- margin) / len(files) - margin
    image_size = min(image_size_x, image_size_y)


    rows = len(files)
    c.setFontSize(10)
    for row in tqdm(range(rows)):
        for col in range(cols):

            if row == 0:
                if col ==1:
                    c.setFontSize(12)
                else:
                    c.setFontSize(12)
                    
                addon = ""
                if scores[col] != "":
                    addon = " (%s)" % str(scores[col])

                c.drawString(margin + col * (image_size + margin), height - margin, 
                             names[col] +  addon)

            x = margin + col * (image_size + margin)
            y = height - margin - (row + 1) * (image_size + margin)

            image_path = os.path.join(active_subdirs[col], files[row])
            # img = Image.open(image_path)
            # img.thumbnail((image_size, image_size), Image.ANTIALIAS)
            # img_name = f"/tmp/temp_{row}_{col}.jpg"
            # img.save(img_name)
            c.drawImage(image_path, x, y, width=image_size, height=image_size)

    c.save()
