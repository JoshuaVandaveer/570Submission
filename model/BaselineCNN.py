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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyCNN(nn.Module):
  def __init__(self, in_channels, in_h, in_w):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, 6, 5)
    size_after_conv1 = in_h - 5 + 1

    self.pool = nn.MaxPool2d(2, 2)
    size_after_max_pool1 = size_after_conv1 // 2 + (size_after_conv1 % 2)

    self.conv2 = nn.Conv2d(6, 16, 5)
    size_after_conv2 = size_after_max_pool1 - 5 + 1

    size_after_max_pool2 = size_after_conv2 // 2 + (size_after_conv2 % 2)
    self.fc1 = nn.Linear(16 *size_after_max_pool2  * size_after_max_pool2, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
if __name__== "__main__":
  # loss function
  import torch.optim as optim
  BaselineCNN = MyCNN(1, 28, 28).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(BaselineCNN.parameters(), lr=0.001, momentum=0.9)

  # count the parameter
  par = sum(p.numel() for p in BaselineCNN.parameters() if p.requires_grad)
  print(f"Total parameters of BaselineCNN: {par}")