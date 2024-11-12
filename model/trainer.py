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

# train the model and print the error periodically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from cyclic_4 import s02_r4
class Trainer():
  def __init__(self, model, criterion, optimizer, train_loader, test_loader):
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.test_loader = test_loader

  def train(self, epochs):
    self.model.train() # we need to set the mode for our model
    total_loss = 0

    len_loader = len(self.train_loader.dataset)
    for e in range(epochs):
      for batch_idx, (images, targets) in enumerate(self.train_loader):
        images, targets = images.to(device), targets.to(device)
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = self.criterion(output, targets) # Here is a typical loss function (negative log likelihood)
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        if batch_idx % 200 == 0: # We visulize our output every 2000 batches
          print(f'Epoch {e}: [{batch_idx*len(images)}/{len_loader}] Loss: {total_loss/200}')
          total_loss = 0.0
    print("Finished training")

  def train_c4(self, epochs):
    self.model.to(device)
    self.model.train() # we need to set the mode for our model
    total_loss = 0

    len_loader = len(self.train_loader.dataset)
    for e in range(epochs):
      for batch_idx, (images, targets) in enumerate(self.train_loader):
        images, targets = images.to(device), targets.to(device)
        b, c, h, w = images.shape
        images = s02_r4(images).reshape(-1, c,h, w).to(device)
        targets = targets.repeat(4).to(device)
        self.optimizer.zero_grad()
        output = self.model(images)
        loss = self.criterion(output, targets) # Here is a typical loss function (negative log likelihood)
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()
        if batch_idx % 20 == 0: # We visulize our output every 10 batches
          print(f'Epoch {e}: [{batch_idx*len(images)}/{len_loader}] Loss: {total_loss/20}')
          total_loss = 0.0
    print("Finished training")

  def test(self):
    self.model.eval() # we need to set the mode for our model
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for images, targets in self.test_loader:
        images, targets = images.to(device), targets.to(device)
        output = self.model(images)
        test_loss += self.criterion(output, targets).item()
        pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
        correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples

    test_loss /= len(self.test_loader.dataset)
    print(f'Test: Avg loss is {test_loss}, Accuracy: {100.*correct/len(self.test_loader.dataset)}%')

  def test_c4(self):
    self.model.eval() # we need to set the mode for our model
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for images, targets in self.test_loader:
        images, targets = images.to(device), targets.to(device)
        b, c, h, w = images.shape
        images = s02_r4(images).reshape(-1, c,h, w).to(device)
        targets = targets.repeat(4).to(device)
        output = self.model(images)
        test_loss += self.criterion(output, targets).item()
        pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
        correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples

    test_loss /= len(self.test_loader.dataset)
    print(f'Test C4: Avg loss is {test_loss}, Accuracy: {100.*correct/(len(self.test_loader.dataset)*4)}%')