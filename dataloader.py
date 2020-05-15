#! /usr/bin/env python3

"""
@authors: Michael Albert (albertm4@wwu.edu)
          Archan Rupela (rupelaa@wwu.edu)
          Ethan Lindell (lindele@wwu.edu)
          River Yearian (yeariar@wwu.edu)
          Jonah Douglas (douglaj8@wwu.edu)

Takes a directory of images and loads them as a dataloader

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""

import torch
import torchvision as tv
import argparse
import sys
import os
import numpy as np

def baseline(path):
    # Determine baseline of data
    currentSize = 0
    maxSize = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                currentSize += 1
        if currentSize > maxSize:
            maxSize = currentSize
            name = filenames;
    return len(name),maxSize

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("tr",help="The root for the training dataset",type=str)
    parser.add_argument("te",help="The root for the testing dataset", type=str)
    parser.add_argument("bs",help="The batch size",type=int)
    return parser.parse_args()

def main(argv):
    # Parse arguments
    args = parse_all_args()

    # Get device to use
    if torch.cuda.is_available():
        dev = "cuda" #GPU
    else:
        dev = "cpu"
    device = torch.device(dev)

    # Load data
    train = tv.datasets.ImageFolder(root=args.tr,transform=None)
    trainloader = torch.utils.data.DataLoader(train,batch_size=args.bs, shuffle=True)
    test = tv.datasets.ImageFolder(root=args.te,transform=None)
    testloader = torch.utils.data.DataLoader(test,batch_size=args.bs, shuffle=False)

    # Create all classes
    classes = list(train.class_to_idx.keys())

    # Determine baseline
    base = baseline(args.te)
    basepercent = base[0]/base[1]*100
    print(basepercent)

if __name__ == "__main__":
    main(sys.argv)
