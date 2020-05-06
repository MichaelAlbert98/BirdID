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
import numpy as np

# class DeepNeuralNet(torch.nn.ModuleList):
#     def __init__(self, D, L, C):
#         """
#         In the constructor we instantiate an arbitrary amount of nn.Linear
#         modules and assign them as a member variable
#         """
#         super(DeepNeuralNet, self).__init__()
#         # Construct Linear for each layer
#         self.linears = torch.nn.ModuleList()
#         input = D
#         for i in L:
#             vals = [int(x) for x in i.split("x")]
#             output = vals[0]
#             for j in range(0,vals[1]):
#                 self.linears.append(torch.nn.Linear(input,output))
#                 input = output
#         output = C
#         self.linears.append(torch.nn.Linear(input,output))
#
#         # Print params
#         for name, param in self.named_parameters():
#             print(name,param.data.shape)
#
#     def forward(self, x, f1):
#         """
#         In the forward function we accept a Tensor of input data and we must
#         return a Tensor of output data. We can use Modules defined in the
#         constructor as well as arbitrary operators on Tensors.
#         """
#         y = x
#         for i,l in enumerate(self.linears):
#             x = l(y)
#             if f1 == "relu":
#                 y = torch.nn.functional.relu(x)
#             elif f1 == "tanh":
#                 y = torch.nn.functional.tanh(x)
#             elif f1 == "sigmoid":
#                 y = torch.nn.functional.logsigmoid(x)
#         return x

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("tr",help="The root for the training dataset",type=str)
    parser.add_argument("te",help="The root for the testing dataset", type=str)
    parser.add_argument("bs",help="The batch size",type=int)
    return parser.parse_args()

# def train(model,train_x,train_y,dev_x,dev_y,N,D,args):
#     criterion = torch.nn.CrossEntropyLoss(reduction='sum')
#     optimizer = None
#     if optimizer == None:
#         if args.opt == "adadelta":
#             optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
#         if args.opt == "adagrad":
#             optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
#         if args.opt == "adam":
#             optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         if args.opt == "rmsprop":
#             optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
#         if args.opt == "sgd":
#             optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
#
#     devN,_ = dev_x.shape
#
#     for epoch in range(args.epochs):
#         # shuffle data once per epoch
#         idx = np.random.permutation(N)
#         train_x = train_x[idx,:]
#         train_y = train_y[idx]
#
#         for update in range(int(np.floor(N/args.mb))):
#             mb_x = train_x[(update*args.mb):((update+1)*args.mb),:]
#             mb_y = train_y[(update*args.mb):((update+1)*args.mb)]
#
#             mb_y_pred = model(mb_x, args.f1) # evaluate model forward function
#             loss      = criterion(mb_y_pred,mb_y) # compute loss
#
#             optimizer.zero_grad() # reset the gradient values
#             loss.backward()       # compute the gradient values
#             optimizer.step()      # apply gradients
#
#             if (update % args.report_freq) == 0:
#                 # eval on dev once per epoch
#                 dev_y_pred     = model(dev_x, args.f1)
#                 _,dev_y_pred_i = torch.max(dev_y_pred,1)
#                 dev_acc        = (dev_y_pred_i == dev_y).sum().data.numpy()/devN
#                 print("%03d.%04d: dev %.3f" % (epoch,update,dev_acc))

def main(argv):
    # parse arguments
    args = parse_all_args()

    # get device to use
    if torch.cuda.is_available():
        dev = "cuda" #GPU
    else:
        dev = "cpu"
    device = torch.device(dev)

    # load data
    train = tv.datasets.ImageFolder(root=args.tr,transform=None)
    trainloader = torch.utils.data.DataLoader(train,batch_size=args.bs, shuffle=True)
    test = tv.datasets.ImageFolder(root=args.te,transform=None)
    testloader = torch.utils.data.DataLoader(test,batch_size=args.bs, shuffle=False)

    # create all classes
    classes = list(train.class_to_idx.keys())
    print(classes[199])

if __name__ == "__main__":
    main(sys.argv)
