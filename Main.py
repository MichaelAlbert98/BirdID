#! /usr/bin/env python3

"""
@authors: Michael Albert (albertm4@wwu.edu)
          Archan Rupela (rupelaa@wwu.edu)
          Ethan Lindell (lindele@wwu.edu)
          River Yearian (yeariar@wwu.edu)
          Jonah Douglas (douglaj8@wwu.edu)

Creates and trains a prototypical network using few shot learning to classify bird species

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.
"""

import torch
import torchvision as tv
import sys
import os
import numpy as np
from parser import parse_all_args
from Utils import euclidean_dist
from protonet import ProtoNet

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
            name = filenames
    return len(name), maxSize

def init_dataset(folder)
    dataset = tv.datasets.ImageFolder(root=folder,transform=None)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset

def init_dataloader(args, mode)
    if 'train' in mode:
        dataset = init_dataset(args.tr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mb, shuffle=True, drop_last=False)
    elif 'valid' in mode:
        dataset = init_dataset(args.va)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mb, shuffle=True, drop_last=False)
    else:
        dataset = init_dataset(args.te)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.mb, shuffle=False, drop_last=False)
    return dataloader

def init_protonet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtoNet().to(device)
    return model


# k-shot (number of examples per class)
# n-way (number of classes)
def get_episode(n,k, dataset)
     C = random.sample(dataset, n)	# randomly sample n classes
	 X = None
     for i in range(len(C)):
		X[i] = random.sample(C[i], 2*k) # randomly sample 2*k (image,label) pairs from each class in C
	 
	 S = None
     for i in range(len(X)):
		S[i] = random.sample(X[i], k) # randomly take k of the (image,label) pairs for each class from X # support
	 
	 Q = None
	 for i in range(len(X)):
		for j in range(len(X[0])):
			if X[i][j] not in S[i]:
				Q[i].append(X[i][j]) # take the remaining k classes' data from X # query set

     return S,Q

		 
def train(my_train, args, my_valid, model)
	#for S,Q in my_train:
		# embed all images in S to produce n prototypes
		# embed all images in Q and compute the posterior probabilities

	# loop over classes
	for _ in range(args.its):
		S,Q = get_episode(args.n, args.k, my_train)
		
         # compute loss for all k*n query images
         # backprop
         # update weights
		 
	# loop over classes

         # compute loss for all k*n query images
         # compute acuracy for the k*n query images
		 
def test()

def eval()
	for S,Q in my_dev:
		# embed all images in S to produce n prototypes
        # embed all images in Q and computer the posterior probabilities

def main():
    # Parse arguments
    args = parse_all_args()

    # # Determine baseline
    # base = baseline(args.te)
    # basepercent = base[0]/base[1]*100
    # print(basepercent)

    # Load data
    train_loader = init_dataloader(args, 'train')
    valid_loader = init_dataloader(args, 'valid')
    test_loader = init_dataloader(args, 'test') 

    # Create model
    model = init_protonet()
    result = train(train_loader, args, valid_loader, model)
	
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = result
    
	print("Testing with current model")
    test(args, test_loader, model)
    
	model.load_state_dict(best_state)
    
	print("Testing with best model")
    test(args, test_loader, model)

if __name__ == "__main__":
    main()
