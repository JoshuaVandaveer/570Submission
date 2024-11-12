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
from cyclic_4 import cifar

# encoder module

##################################################
class C_4_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 4) / math.sqrt(4 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 4).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = self.weight[..., [3, 0, 1, 2]]
        weight[::, 2, ...] = self.weight[..., [2, 3, 0, 1]]
        weight[::, 3, ...] = self.weight[..., [1, 2, 3, 0]]
        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 4, self.in_channels * 4, 1, 1), stride=1,
                                       padding=0)
        return x

##################################################
class C_4_1x1_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_1x1_, self).__init__()
        self.net = nn.Conv3d(in_channels, out_channels, 1, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.net(x.view(b, c // 4, 4, h, w)).reshape(b, self.out_channels * 4, h, w)
        return x

##################################################
class C_4_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 4, 3, 3) / math.sqrt(4 * in_channels * 9 / 2)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 4, 3, 3).to(x.device)
        weight[::, 0, ...] = self.weight
        weight[::, 1, ...] = torch.rot90(self.weight[..., [3, 0, 1, 2], ::, ::], 1, [3, 4])
        weight[::, 2, ...] = torch.rot90(self.weight[..., [2, 3, 0, 1], ::, ::], 2, [3, 4])
        weight[::, 3, ...] = torch.rot90(self.weight[..., [1, 2, 3, 0], ::, ::], 3, [3, 4])
        x = torch.nn.functional.conv2d(x, weight.reshape(self.out_channels * 4, self.in_channels * 4, 3, 3))
        return x

##################################################
class C_4_BN(nn.Module):
    def __init__(self, in_channels):
        super(C_4_BN, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        return self.bn(x.reshape(b, c // 4, 4, h, w)).reshape(x.size())

##################################################
class C_4_Pool(nn.Module):
    def __init__(self):
        super(C_4_Pool, self).__init__()
        self.pool = nn.MaxPool3d((4, 1, 1), (4, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return self.pool(x.reshape(b, c // 4, 4, h, w)).squeeze(2)

##################################################
class E4_C4(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 reduction_ratio=2,
                 groups=1
                 ):

        super(E4_C4, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.group_channels = groups
        self.groups = self.out_channels // self.group_channels
        self.dim_g = 4

        self.v = nn.Sequential(C_4_1x1(in_channels, out_channels))
        self.conv1 = nn.Sequential(C_4_1x1(in_channels, int(in_channels // reduction_ratio)),
                                    nn.GroupNorm(int(in_channels // reduction_ratio),int(in_channels // reduction_ratio)*4), nn.ReLU())
        self.conv2 = nn.Sequential(C_4_1x1_(int(in_channels // reduction_ratio), kernel_size ** 2 * self.groups))

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride=1)

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, 4, h, w)
        weight[::, ::, ::, ::, 1, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 1, ::, ::], 1, [2, 3])
        weight[::, ::, ::, ::, 2, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 2, ::, ::], 2, [2, 3])
        weight[::, ::, ::, ::, 3, ::, ::] = torch.rot90(weight[::, ::, ::, ::, 3, ::, ::], 3, [2, 3])
        weight = weight.reshape(b, self.groups, self.kernel_size ** 2, 4, h, w).unsqueeze(2).transpose(3, 4)
        x = self.v(x)
        out = self.unfold(x).view(b, self.groups, self.group_channels, 4, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=4).view(b, self.out_channels * 4, h, w)
        return out

##################################################
class C_4_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_4_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight = torch.randn(out_channels, in_channels, 3, 3) / math.sqrt(9 * in_channels / 2)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, input):
        weight = torch.zeros(self.out_channels, 4, self.in_channels, 3, 3).to(input.device)
        weight[::, 0] = self.weight
        weight[::, 1] = torch.rot90(self.weight[::], 1, [2, 3])
        weight[::, 2] = torch.rot90(self.weight[::], 2, [2, 3])
        weight[::, 3] = torch.rot90(self.weight[::], 3, [2, 3])
        out = nn.functional.conv2d(input, weight.reshape(self.out_channels * 4, self.in_channels, 3, 3), padding=1)
        return out
    

if __name__ == "__main__":
    Ciphar = cifar()
    print(f"Training size is {len(Ciphar.train_loader)}")
    print(f"Testing size is {len(Ciphar.test_loader)}")

    dataiter = iter(Ciphar.train_loader)
        
    images, labels = next(dataiter)
    images_c4 = images.repeat(1, 4, 1, 1)
    print(f"Inputs: images={images.shape}, label={labels.shape}")

    TestC4_1x1 = C_4_1x1(3,1)
    out_a = TestC4_1x1(images_c4)
    print(f"C_4_1x1 outputs: {out_a.shape}")

    TestC4_1x1_ = C_4_1x1_(3,2)
    out_b = TestC4_1x1_(images_c4)
    print(f"C_4_1x1_ outputs: {out_b.shape}")

    TestC4_3x3 = C_4_3x3(3,3)
    out_c = TestC4_3x3(images_c4)
    print(f"TestC4_3x3 outputs: {out_c.shape}")

    TestEC4 = E4_C4(3,4,5, reduction_ratio=2, groups=1)
    out_d = TestEC4(images_c4)
    print(f"TestEC4 outputs: {out_d.shape}")