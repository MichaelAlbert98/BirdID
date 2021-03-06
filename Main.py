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
import matplotlib.pyplot as plt
import numpy as np
from Parser import parse_all_args
from Utils import euclidean_dist
from Model import vgg11
from Batch_Sampler import PrototypicalBatchSampler

def init_dataset(path):
    dataset = tv.datasets.ImageFolder(root=path,transform=tv.transforms.ToTensor())
    return dataset

def init_sampler(n, k, eps, dataset, name):
    classes_per_it = n
    num_samples = 2*k
    return PrototypicalBatchSampler(dataset=dataset,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    episodes=eps,
                                    name=name)


def init_dataloader(args, mode):
    if mode == 'train':
        dataset = init_dataset(args.tr)
        sampler = init_sampler(args.n, args.k, args.eps, dataset, "train")
    elif mode == 'valid':
        dataset = init_dataset(args.va)
        sampler = init_sampler(args.n, args.k, args.eps, dataset, "valid")
    elif mode == 'test':
        dataset = init_dataset(args.te)
        sampler = init_sampler(args.n, args.k, args.eps, dataset, "test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

def init_model(pretrain):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg11(pretrained=pretrain).to(device)
    return model
	 
def train(model, tr_loader, va_loader, args):
    f = open("log.txt", "a+")
    x = []
    y1 = []
    y2 = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

	# loop over iterations
    for iteration in range(args.its):
        # loop over episodes
        episodes = iter(tr_loader)
        number = 0
        for epi in episodes:
            number += 1
            print(number)
            epi[0] = epi[0].reshape(args.n, args.k*2, 224, 224, 3)
            epi[0] = epi[0].permute(0,1,4,2,3)
            S = epi[0][:, :args.k, :, :, :].to(device)
            Q = epi[0][:, args.k:, :, :, :].to(device)
            Q = Q.reshape(args.n*args.k, 3, 224, 224)

            # embed S/Q
            embedS = torch.zeros(args.n, 4096)         # is there a way to not hardcode this?
            embedQ = torch.zeros(args.n*args.k, 4096)  # '                                  '
            for i in range(S.size(0)):
                s = slice(i * args.k, (i + 1) * args.k)
                allS = model(S[i])
                embedS[i] = torch.sum(allS, 0)/len(S[i]) # take average of S to create n prototypes
                embedQ[s] = model(Q[s])

            # find euclidean distance to each S for each Q
            euclid = torch.zeros(args.n*args.k, args.n)
            for i in range(len(embedQ)):
                    for j in range(len(embedS)):
                        euclid[i][j] = -1*euclidean_dist(embedQ[i], embedS[j])

            # compute loss for all k*n query images
            classes = torch.arange(0, args.n, 1)
            C = torch.repeat_interleave(classes, args.k)
            loss = criterion(euclid, C)

            # backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate on dev set once per iteration
        # loop over episodes
        avg_arr = []
        episodes = iter(va_loader)
        for epi in episodes:
            epi[0] = epi[0].reshape(args.n, args.k*2, 224, 224, 3)
            epi[0] = epi[0].permute(0,1,4,2,3)
            S = epi[0][:, :args.k, :, :, :].to(device)
            Q = epi[0][:, args.k:, :, :, :].to(device)
            Q = Q.reshape(args.n*args.k, 3, 224, 224)
        
            # embed S/Q
            embedS = torch.zeros(5, 4096)             # is there a way to not hardcode this?
            embedQ = torch.zeros(args.n*args.k, 4096) # '                                  '
            for i in range(S.size(0)):
                s = slice(i * args.k, (i + 1) * args.k)
                allS = model(S[i])
                embedS[i] = torch.sum(allS, 0)/len(S[i]) # take average of S to create n prototypes
                embedQ[s] = model(Q[s])

            # find euclidean distance to each S for each Q
            euclid = torch.zeros(args.n*args.k, args.n)
            for i in range(len(embedQ)):
                for j in range(len(embedS)):
                    euclid[i][j] = -1*euclidean_dist(embedQ[i], embedS[j])

            # classify queries to nearest prototype
            _, idxs = torch.max(euclid, 1)
            tot = 0
            acc = 0
            for i in range(len(idxs)):
                    tot += 1
                    if idxs[i] == i//args.k:
                        acc += 1
            avg_arr.append(acc/tot)

        avg_arr = np.asarray(avg_arr)
        mean = avg_arr.mean()
        std = avg_arr.std()
        output = ("Iteration %04d: Mean - %.3f Std - %.3f \n" % (iteration, mean, std))
        f.write(output)
        x.append(iteration)
        y1.append(mean)
        y2.append(std)

    f.close()
    plt.ylim(0,1)
    plt.plot(x,y1, label='Mean')
    plt.plot(x,y2, label="Std")
    plt.xlabel('Iterations')
    plt.ylabel('Percentage')
    plt.title('Few-shot Learning')
    plt.legend()
    plt.savefig("image.png")

		 
def test(model, te_loader, args):
    # evaluate on test set at end of training
    # loop over episodes
    f = open("log.txt", "a+")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_arr = []
    episodes = iter(te_loader)
    for epi in episodes:
        epi[0] = epi[0].reshape(args.n, args.k*2, 224, 224, 3)
        epi[0] = epi[0].permute(0,1,4,2,3)
        S = epi[0][:, :args.k, :, :, :].to(device)
        Q = epi[0][:, args.k:, :, :, :].to(device)
        Q = Q.reshape(args.n*args.k, 3, 224, 224)
    
        # embed S/Q
        embedS = torch.zeros(5, 4096)             # is there a way to not hardcode this?
        embedQ = torch.zeros(args.n*args.k, 4096) # '                                  '
        for i in range(S.size(0)):
            s = slice(i * args.k, (i + 1) * args.k)
            allS = model(S[i])
            embedS[i] = torch.sum(allS, 0)/len(S[i]) # take average of S to create n prototypes
            embedQ[s] = model(Q[s])

        # find euclidean distance to each S for each Q
        euclid = torch.zeros(args.n*args.k, args.n)
        for i in range(len(embedQ)):
            for j in range(len(embedS)):
                euclid[i][j] = -1*euclidean_dist(embedQ[i], embedS[j])

        # classify queries to nearest prototype
        _, idxs = torch.max(euclid, 1)
        tot = 0
        acc = 0
        for i in range(len(idxs)):
                tot += 1
                if idxs[i] == i//5:
                    acc += 1
        avg_arr.append(acc/tot)

    avg_arr = np.asarray(avg_arr)
    mean = avg_arr.mean()
    std = avg_arr.std()
    output = ("Testing: Mean - %.3f Std - %.3f \n" % (mean, std))
    f.write(output)
    f.close()

def main():
    # Parse arguments
    args = parse_all_args()

    # Load data
    train_loader = init_dataloader(args, "train")
    valid_loader = init_dataloader(args, "valid")
    test_loader = init_dataloader(args, "test") 

    # Create and test model
    model = init_model(args.pre)
    train(model, train_loader, valid_loader, args)
    test(model, test_loader, args)

if __name__ == "__main__":
    main()
