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

# model imports
from E4_functions import C_4_3x3, C_4_BN, C_4_Conv
from cyclic_4 import cifar

class C4Basic(nn.Module):
    def __init__(self, in_h, in_w, in_channels, kernel, reduction, groups):
        super().__init__()

        self.conv1=C_4_Conv(in_channels, 16)

        self.forward_function1 = nn.Sequential(
            C_4_3x3(16, 16),
            C_4_BN(16),
            nn.ReLU(inplace=True),
            C_4_3x3(16, 16),
            C_4_BN(16)
        )

        self.forward_function2 = nn.Sequential(
            C_4_3x3(16, 16),
            C_4_BN(16),
            nn.ReLU(inplace=True),
            C_4_3x3(16, 4),
            C_4_BN(4)
        )

        self.pool = nn.MaxPool2d(2, 2)
        h = (in_h - 8) // 2 + ( (in_h - 8) % 2 )
        w = (in_w - 8) // 2 + ( (in_w - 8) % 2 )

        in_nodes = 4 * 4 * h * w
        self.fc1 = nn.Linear(in_nodes, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(self.forward_function1(x) )
        x = nn.ReLU(inplace=True)(self.forward_function2(x) )
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    Ciphar = cifar()
    print(f"Training size is {len(Ciphar.train_loader)}")
    print(f"Testing size is {len(Ciphar.test_loader)}")
    
    dataiter = iter(Ciphar.train_loader)
    images, labels = next(dataiter)
    images_c4 = images.repeat(1, 4, 1, 1)
    print(f"Inputs: images={images.shape}, label={labels.shape}")

    TestC4Basic = C4Basic(
                        in_h=32,
                        in_w=32,
                        in_channels=3,
                        kernel=5,
                        reduction=1,
                        groups=1)
    out_e = TestC4Basic(images)
    print(f"TestC4Basic outputs: {out_e.shape}")

    par = sum(p.numel() for p in TestC4Basic.parameters() if p.requires_grad)
    print(f"Total parameters of TestC4Basic: {par}")