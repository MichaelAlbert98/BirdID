#! /usr/bin/env python3

"""
@authors: Michael Albert (albertm4@wwu.edu)

Augments a dataset of images with random affine + color jitter transforms

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.
"""

#import torch
import torchvision as tv
from PIL import Image
import sys
import os
#import numpy as np

def main(path):
    transform = tv.transforms.Compose([
                    tv.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
                    tv.transforms.RandomAffine(20),
                ])
    # Walk over all files and augment
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                img = Image.open(fp)
                base,ext = os.path.splitext(fp)
                for i in range(3):
                    new_img = transform(img)
                    new_img.save("%saug%d%s" % (base,i,ext))

if __name__ == "__main__":
    main(sys.argv[1])
