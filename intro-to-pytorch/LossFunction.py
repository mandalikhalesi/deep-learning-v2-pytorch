import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Define a transformer to normalise data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download then load MNIST training dataset
print("\nDownloading training data...\n") 
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
print("\nLoading training data...\n") 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print("\nView of downloaded trainloader data:\n", trainloader) 

# Make trainloader iterable
dataiter = iter(trainloader)
print("\nView of dataiter\n", dataiter) 

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10)
                      )
print ("\nThis is x (784 inputs to 128-relu to 64-relu to 10-(pre-activation) outputs):\n", model)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Get data from iterable dataiter
images, labels = next(dataiter)
print("\nType of images is:\n", type(images))
print("\nimages.shape is:\n", images.shape)
print("\nlabels.shape is:\n", labels.shape)

# Flatten images
images = images.view(images.shape[0], -1)
print("\nFlattened images:\n", images)
print("\nFlattened images.shape is:\n", images.shape)

# Forward pass, get logits
print ("\nStarting forward pass...\n")
logits = model(images)
print ("\nResulting logits are:\n", logits)

# Calculate loss with logits and labels
loss = criterion(logits, labels)
print ("\nResulting loss is:\n", loss)
