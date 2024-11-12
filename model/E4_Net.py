import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from E4_functions import E4_C4, C_4_BN, C_4_Conv, C_4_Pool
from cyclic_4 import cifar


class E4_net(nn.Module):
    def __init__(self, in_channels=1, kernel_size=5, groups=8, reduction_ratio=1, drop=0.2):
        super(E4_net, self).__init__()

        # lifting convolution
        self.conv1=C_4_Conv(in_channels, 16)

        # pooling through conv for dimention reductions
        self.conv2=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv3=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv4=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv5=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv6=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv7=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.pool=nn.MaxPool2d(2,2)

        self.bn1=C_4_BN(16)
        self.bn2=C_4_BN(16)
        self.bn3=C_4_BN(16)
        self.bn4=C_4_BN(16)
        self.bn5=C_4_BN(16)
        self.bn6=C_4_BN(16)
        self.bn7=C_4_BN(16)

        self.drop=nn.Dropout(drop)


        self.group_pool=C_4_Pool()
        self.global_pool=nn.AdaptiveMaxPool2d(1)

        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(16, 10),
        )

    def forward(self, x):
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)
        x=self.drop(torch.relu(self.conv3(x)))
        x=self.drop(torch.relu(self.conv4(x)))
        x=self.pool(x)
        x=self.drop(torch.relu(self.bn5(self.conv5(x))))
        x=self.drop(torch.relu(self.conv6(x)))
        x=self.drop(torch.relu(self.conv7(x)))
        x=self.group_pool(x)
        x=self.global_pool(x).reshape(x.size(0),-1)
        x=self.fully_net(x)
        return x
    
if __name__ == "__main__":
    Ciphar = cifar()
    print(f"Training size is {len(Ciphar.train_loader)}")
    print(f"Testing size is {len(Ciphar.test_loader)}")
    
    dataiter = iter(Ciphar.train_loader)
    images, labels = next(dataiter)
    images_c4 = images.repeat(1, 4, 1, 1)
    print(f"Inputs: images={images.shape}, label={labels.shape}")

    TestE4_net = E4_net(in_channels=3,kernel_size=5, groups=8, reduction_ratio=1, drop=0.2)
    out_f = TestE4_net(images)
    print(f"TestE4_net outputs: {out_f.shape}")

    par = sum(p.numel() for p in TestE4_net.parameters() if p.requires_grad)
    print(f"Total parameters of TestE4_net: {par}")