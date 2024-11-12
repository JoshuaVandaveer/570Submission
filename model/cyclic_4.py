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

def s02_r4(x):
  b, c, h, w = x.shape
  elements = torch.zeros((b, 4, c, h, w))
  # first element is the original all others are rotated by 90 degress along the height/width dim (2,3)
  elements[::, 0, ...] = x
  elements[::, 1, ...] = torch.rot90(x, 1, [2,3])
  elements[::, 2, ...] = torch.rot90(x, 2, [2,3])
  elements[::, 3, ...] = torch.rot90(x, 3, [2,3])
  return elements

# create the cifar10 datsets, transforms, group elements, and plot
class cifar():
  def __init__(self):
    self.mean = np.array([0.49139968, 0.48215827 ,0.44653124])
    self.std = np.array([0.24703233, 0.24348505, 0.26158768])
    self.t = T.Compose( [T.ToTensor(),    # convert images to tensor form (pushes channel dimentions to beginning)
             T.Normalize( self.mean,      # normalize with the known mean
                          self.std) ])    # normalize with the known standard deviation
    self.inv_t = T.Normalize(mean=(-self.mean/self.std), std=(1/self.std))

    # set the batch size for training
    self.batch_size = 4

    # load the training set and create a data loader for the training set
    self.train_data = torchvision.datasets.CIFAR10(root='./data', download=True,train=True, transform=self.t)
    self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size,
                                              shuffle=True, num_workers=2)

    self.test_data = torchvision.datasets.CIFAR10(root='./data', download=True, transform=self.t, train=False)
    self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=2)

    self.classes = ( 'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck' )

  def transform_image(self, img):
    img = self.inv_t(img)    # unormalize
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    return img

  def plot_rotations(self, images):
    fig, axes = plt.subplots(self.batch_size, 4, figsize=(8,8))
    for i  in range(self.batch_size):
      for j in range(4):
        image = images[i, j]
        image = self.transform_image(image)
        ax = axes[i, j]
        ax.imshow(image)
        ax.axis('off')

class mnist():
  def __init__(self):
    self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.1307,),(0.3081,))])

    self.train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=self.transform)
    self.test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=self.transform)

    self.batch_size = 4
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

  def transform_image(self, img):
    img = img[0]
    return img

  def plot_rotations(self, images):
    fig, axes = plt.subplots(self.batch_size, 4, figsize=(8,8))
    for i  in range(self.batch_size):
      for j in range(4):
        image = images[i, j]
        image = self.transform_image(image)
        ax = axes[i, j]
        ax.imshow(image)
        ax.axis('off')

if __name__ == "__main__":
    # test the mnist dataset loader and cyclic group
    Mnist = mnist()
    print(f"Training size is {len(Mnist.train_loader)}")
    print(f"Testing size is {len(Mnist.test_loader)}")

    dataiter = iter(Mnist.train_loader)
    images, labels = next(dataiter)

    print(f"Images before rotation: {images.shape}")
    images = s02_r4(images)
    print(f"Images after rotation: {images.shape}")

    Mnist.plot_rotations(images)

    # test the cifar data loader and cyclic group
    Ciphar = cifar()
    print(f"Training size is {len(Ciphar.train_loader)}")
    print(f"Testing size is {len(Ciphar.test_loader)}")

    dataiter = iter(Ciphar.train_loader)
    images, labels = next(dataiter)

    print(f"Images before rotation: {images.shape}")
    images = s02_r4(images)
    print(f"Images after rotation: {images.shape}")

    Ciphar.plot_rotations(images)

    # test expansion of the rotations into the batch size
    dataiter = iter(Mnist.train_loader)
    images, labels = next(dataiter)

    print(f"Images before rotation: {images.shape}")
    images = s02_r4(images)
    print(f"Images after rotation: {images.shape}")

    b, r, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    labels = np.repeat(labels,r)

    print(f"Images after expansion into batch dimention: {images.shape}")
    print(f"Labels after expansion into for rotations: {labels.shape}")


    fig, axes = plt.subplots(images.shape[0], figsize=(15,15))
    for i in range(images.shape[0]):
        plt.imshow(images[i][0])
        ax = axes[i]
        ax.imshow(images[i][0])
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')

    plt.subplots_adjust(wspace=0.4, hspace=2)