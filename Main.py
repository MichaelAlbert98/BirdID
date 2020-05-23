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
    return dataset

def init_protonet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProtoNet().to(device)
    return model

# k-shot (number of examples per class)
# n-way (number of classes)
# note: needs to be updated to use permute to get S/Q instead of current method
def get_episode(n,k, dataset)
     C = random.sample(dataset, n)	# randomly sample n classes
	 X = None
     for i in range(len(C)):
		X[i] = random.sample(C[i], 2*k) # randomly sample 2*k (image,label) pairs from each class in C
	 
	 S = None
     for i in range(len(X)):
		S[i] = random.sample(X[i], k) # randomly take k of the (image,label) pairs for each class from X # support set
	 
	 Q = None
	 for i in range(len(X)):
		for j in range(len(X[0])):
			if X[i][j] not in S[i]:
				Q[i].append(X[i][j]) # take the remaining k classes' data from X # query set

     return S,Q

		 
def train(model, train, valid, args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

	# loop over iterations
	for _ in range(args.its):
        # loop over episodes
        for _ in range(args.eps)
		    S,Q = get_episode(args.n, args.k, train)

            # embed S/Q
            for i in range(len(S)):
                allS = model(S[i])
                embedS[i] = torch.sum(allS, 1)/len(S[i]) # take average of S to create n prototypes
                embedQ[i] = model(Q[i])
            embedQ2 = embedQ.reshape(len(embedQ)*len(embedQ[0]))

            # find euclidean distance to each S for each Q
            for i in range(len(embedQ2)):
                for j in range(len(embedS)):
                    euclid[i][j] = -1*euclidean_dist(embedQ2[i], embedS[j])

            # compute loss for all k*n query images
            loss = 0
            for i in range len(euclid):
                loss += (1/args.n*args.k)*criterion(euclid[i], i//args.n)

            # backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

		 
def test()

def eval()
	for S,Q in my_dev:
		# embed all images in S to produce n prototypes
        # embed all images in Q and compute the posterior probabilities

def main():
    # Parse arguments
    args = parse_all_args()

    # # Determine baseline
    # base = baseline(args.te)
    # basepercent = base[0]/base[1]*100
    # print(basepercent)

    # Load data
    train = init_dataset(args.tr)
    valid = init_dataset(args.va)
    test = init_dataset(args.te) 

    # Create model
    model = init_protonet()
    result = train(model, train, valid, args)
	
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = result
    
	print("Testing with current model")
    test(args, test_loader, model)
    
	model.load_state_dict(best_state)
    
	print("Testing with best model")
    test(args, test_loader, model)

if __name__ == "__main__":
    main()
