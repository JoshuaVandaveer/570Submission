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

from E4_functions import E4_C4, C_4_BN, C_4_Conv, C_4_Pool, C_4_3x3
from cyclic_4 import cifar
from subsample import EquivariantSubsample

class E4_Pooling(nn.Module):
    def __init__(self, in_w, in_channels=1, kernel_size=5, groups=8, reduction_ratio=1, drop=0.2):
        super(E4_Pooling, self).__init__()

        # lifting convolution
        self.liftingConv=C_4_Conv(in_channels, 16)

        # conv 3x3 + equivariant pooling
        self.conv3x3_1 = C_4_3x3(16,16)
        self.pool1 = EquivariantSubsample(reduction=(2,2))
        self.conv3x3_2 = C_4_3x3(16,16)
        self.pool2 = EquivariantSubsample(reduction=(2,2))

        # E4 network
        self.conv1=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv2=E4_C4(16, 16, kernel_size, reduction_ratio=reduction_ratio, groups=groups)
        self.conv3=E4_C4(16, 16, kernel_size, reduction_ratio=1, groups=groups)

        self.ReductionConv1=E4_C4(4, 2, kernel_size, reduction_ratio=1, groups=1)
        self.ReductionConv2=E4_C4(2, 1, kernel_size, reduction_ratio=1, groups=1)

        # norm functions
        self.bn1=C_4_BN(16)
        self.bn2=C_4_BN(16)
        self.bn3=C_4_BN(16)

        self.group_pool=C_4_Pool()
        #self.global_pool=nn.AdaptiveMaxPool2d()
        final_size = ((((in_w-2)//2 ) - 2) // 2)**2
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(final_size*4, 36),
            torch.nn.Linear(36, 10)
        )


    def forward(self, x):
        # 3x3 conv 1
        x=torch.relu(self.liftingConv(x))
        x = self.conv3x3_1(x)
        p1 = self.pool1.get_p(x)
        x = self.pool1(x, p1)

        # 3x3 conv 2
        x = self.conv3x3_2(x)
        p2 = self.pool2.get_p(x)
        x = self.pool1(x,p2)


        # x=torch.relu(self.bn2(self.conv2(x)))
        # x=self.pool(x)

        x=torch.relu(self.bn1(self.conv1(x)))
        x=torch.relu(self.bn2(self.conv2(x)))
        x=torch.relu(self.bn3(self.conv3(x)))

        #x = self.ReductionConv2(x)

        x=self.group_pool(x)
        x = self.ReductionConv1(x)
        x = self.ReductionConv2(x)

        x = torch.flatten(x, 1)
        x=self.fully_net(x)
        return x
    
if __name__ == "__main__":
    # loss function
    import torch.optim as optim
    E4_Pooling_net = E4_Pooling(in_w=32, in_channels=3, kernel_size=5, groups=8, reduction_ratio=1, drop=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(E4_Pooling_net.parameters(), lr=0.001, momentum=0.9)

    # print total number of parameters
    par = sum(p.numel() for p in E4_Pooling_net.parameters() if p.requires_grad)
    print(f"Total parameters of E4_Pooling: {par}")