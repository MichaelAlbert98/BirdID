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
import random
import numpy as np
from Parser import parse_all_args
from Utils import euclidean_dist
from Model import vgg11
from Batch_Sampler import PrototypicalBatchSampler
# debugging
import pdb

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

def init_dataset(args, path):
    dataset = tv.datasets.ImageFolder(root=path,transform=tv.transforms.ToTensor())
    return dataset

def init_sampler(args, dataset):
    classes_per_it = args.n
    num_samples = 2*args.k
    return PrototypicalBatchSampler(dataset=dataset,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=args.its)


def init_dataloader(args, path):
    dataset = init_dataset(args, path)
    sampler = init_sampler(args, dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

def init_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg11().to(device)
    return model

# k-shot (number of examples per class)
# n-way (number of classes)
# note: needs to be updated to use permute to get S/Q instead of current method
# def get_episode(n,k, dataset):

#     #debugging
#     pdb.set_trace()
#     num_classes = len(dataset.classes)
#     C = torch.randperm(num_classes)[:5] # randomly sample n classes
#     X = torch.LongTensor(n)
#     for i in range(len(C)):
# 	    X[i] =  # random.sample(C[i], 2*k) # randomly sample 2*k (image,label) pairs from each class in C
	 
#     S = None
#     for i in range(len(X)):
#        S[i] = random.sample(X[i], k) # randomly take k of the (image,label) pairs for each class from X (support set)
	 
#     Q = None
#     for i in range(len(X)):
#         for j in range(len(X[0])):
#             if X[i][j] not in S[i]:
#                 Q[i].append(X[i][j]) # take the remaining k classes' data from X (query set)

#     return S,Q

		 
def train(model, tr_loader, va_loader, args):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

	# loop over iterations
    for _ in range(args.its):
        # loop over episodes
        episodes = iter(tr_loader)
        for epi in episodes:
            epi[0] = epi[0].reshape(args.n, args.k*2, 224, 224, 3)
            S = epi[0][:, :args.n, :, :, :]
            Q = epi[0][:, args.n:, :, :, :]
            S = S.transpose(2,4)
            S = S.transpose(3,4)
            Q = Q.transpose(2,4)
            Q = Q.transpose(3,4)
            # embed S/Q
            embedS = torch.zeros(args.n, 4096)         # is there a way to not hardcode this?
            embedQ = torch.zeros(args.n, args.n, 4096) # '                                  '
            for i in range(S.size(0)):
                allS = model(S[i])
                embedS[i] = torch.sum(allS, 0)/len(S[i]) # take average of S to create n prototypes
                embedQ[i] = model(Q[i])

            # find euclidean distance to each S for each Q
            euclid = torch.zeros(args.n, args.k, args.n)
            for i in range(len(embedQ)):
                for j in range(len(embedQ[0])):
                    for k in range(len(embedS)):
                        euclid[i][j][k] = -1*euclidean_dist(embedQ[i][j], embedS[k])

            # compute loss for all k*n query images
            loss = 0
            for i in range(len(euclid)):
                for j in range(len(euclid[0])):
                    for k in range(len(euclid[0][0])):
                        loss += (1/args.n*args.k)*criterion(euclid[i][j:j+1], torch.tensor([k]))

            #pdb.set_trace() debug
            # backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

		 
def test():
    return None

def eval():
    return None
    #for S,Q in my_dev:
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
    train_loader = init_dataloader(args, args.tr)
    valid_loader = init_dataloader(args, args.va)
    test_loader = init_dataloader(args, args.te) 

    # Create model
    model = init_model()
    train(model, train_loader, valid_loader, args)
	
    #best_state, best_acc, train_loss, train_acc, val_loss, val_acc = result
    
	# print("Testing with current model")
    # test(args, test_loader, model)
    
	# model.load_state_dict(best_state)
    
	# print("Testing with best model")
    # test(args, test_loader, model)

if __name__ == "__main__":
    main()
