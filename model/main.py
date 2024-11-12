# import libraries
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import matplotlib.pyplot as plt

# import trainers and data
from cyclic_4 import mnist
from trainer import Trainer

# import models
from E4_Net import E4_net
from C4Basic import C4Basic
from BaselineCNN import MyCNN
from ES4_Net import E4_Pooling

# import device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
if __name__ == "__main__":
    Mnist = mnist()

    if sys.argv[1] == "Vanilla":
        print("Training Vanilla CNN")
        BaselineCNN = MyCNN(1, 28, 28).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(BaselineCNN.parameters(), lr=0.001, momentum=0.9)
        # count the parameter
        par = sum(p.numel() for p in BaselineCNN.parameters() if p.requires_grad)
        print(f"Total parameters of BaselineCNN: {par}")
        BaselineCNN_mnist = Trainer(BaselineCNN, criterion, optimizer, Mnist.train_loader, Mnist.test_loader)
        
        if sys.argv[2] == "-e": 
            print("Baseline CNN training with equivarience")
            BaselineCNN_mnist.train_c4(2)
        elif sys.argv[2] == "-ne":
            print("Baseline CNN training without equivarience")
            BaselineCNN_mnist.train(2)
        
        BaselineCNN_mnist.test()
        BaselineCNN_mnist.test_c4()

    elif sys.argv[1] == "Basic-Net":
        print("Training Basic Net")
        C4BasicNet = C4Basic(
                      in_h=28,
                      in_w=28,
                      in_channels=1,
                      kernel=5,
                      reduction=2,
                      groups=1).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(C4BasicNet.parameters(), lr=0.001, momentum=0.9)

        # print total number of parameters
        par = sum(p.numel() for p in C4BasicNet.parameters() if p.requires_grad)
        print(f"Total parameters of C4BasicNet: {par}")

        C4BasicNet_mnist = Trainer(C4BasicNet, criterion, optimizer, Mnist.train_loader, Mnist.test_loader)
        if sys.argv[2] == "-e": 
            print("C4Basic CNN training with equivarience")
            C4BasicNet_mnist.train_c4(2)
        elif sys.argv[2] == "-ne":
            print("C4Basic CNN training without equivarience")
            C4BasicNet_mnist.train(2)
        
        C4BasicNet_mnist.test()
        C4BasicNet_mnist.test_c4()

    elif sys.argv[1] == "ES4-Net":
        print("Training ES4-Net")
        E4_Pooling_net = E4_Pooling(in_w=28,in_channels=1, kernel_size=5, groups=8, reduction_ratio=1, drop=0.2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(E4_Pooling_net.parameters(), lr=0.001, momentum=0.9)

        # print total number of parameters
        par = sum(p.numel() for p in E4_Pooling_net.parameters() if p.requires_grad)
        print(f"Total parameters of E4_Pooling_net: {par}")

        E4_Pooling_net_mnist = Trainer(E4_Pooling_net, criterion, optimizer, Mnist.train_loader, Mnist.test_loader)
        E4_Pooling_net_mnist.test()
        E4_Pooling_net_mnist.test_c4()
        if sys.argv[2] == "-e": 
            print("ES4_Net CNN training with equivarience")
            E4_Pooling_net_mnist.train_c4(2)
        elif sys.argv[2] == "-ne":
            print("ES4_Net CNN training without equivarience")
            E4_Pooling_net_mnist.train(2)
        E4_Pooling_net_mnist.test()
        E4_Pooling_net_mnist.test_c4()

    elif sys.argv[1] == "E4-Net":
        print("Training E4-Net")
        TestE4_net = E4_net(kernel_size=5, groups=8, reduction_ratio=1, drop=0.2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(TestE4_net.parameters(), lr=0.001, momentum=0.9)

        # print total number of parameters
        par = sum(p.numel() for p in TestE4_net.parameters() if p.requires_grad)
        print(f"Total parameters of E4_net: {par}")
        E4_net_mnist = Trainer(TestE4_net, criterion, optimizer, Mnist.train_loader, Mnist.test_loader)

        if sys.argv[2] == "-e": 
            print("E4-Net CNN training with equivarience")
            E4_net_mnist.train_c4(2)
        elif sys.argv[2] == "-ne":
            print("E4-Net CNN training without equivarience")
            E4_net_mnist.train(2)
        E4_net_mnist.test()
        E4_net_mnist.test_c4()