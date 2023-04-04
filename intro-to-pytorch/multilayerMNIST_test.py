# Import necessary packages

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

from numpy import exp

# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
print("\nDownloading training data...\n") 
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
print("\nLoading training data...\n") 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print("\nView of downloaded trainloader data:\n", trainloader) 

dataiter = iter(trainloader)
print("\nView of dataiter\n", dataiter) 

#images, labels = dataiter.next() - mistake in code as throws error, below is correct notation reading the iterable dataiter:
images, labels = next(dataiter)
print("\nType of images is:\n", type(images))
print("\nImages.shape is:\n", images.shape)
print("\nLabels.shape is:\n", labels.shape)

def activation(x):
    #Sigmoid activation function
    return 1 / (1 + torch.exp(-x))

    #ReLU activation function
    #return max(0, x)

# Flatten the input images

features = images.view(images.shape[0], -1)
print("\nThese are features:\n", features) 

# Features are 3 random normal variables
#features = torch.randn((1, 3))

# Define the size of each layer in our network
#n_input = features.shape[1]     # Number of input units, must match number of input features
n_input = 784                    # Number of input units, must match number of input features
n_hidden = 256                   # Number of hidden units 
n_output = 10                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
print("\nThese are weights (input to hidden):\n", W1) 
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)
print("\nThese are weights (hidden to output):\n", W2) 

# and bias terms for hidden and output layers
B1 = torch.randn(n_hidden)
print("\nThese are biases (input to hidden):\n", B1) 

B2 = torch.randn(n_output)
print("\nThese are biases (hidden to output):\n", B2) 

# 1st activation function (input to hidden)

features_times_weights_inputtohidden = torch.mm(features, W1)
print("\nThis is features mm weights (input to hidden):\n", features_times_weights_inputtohidden) 

y = activation(features_times_weights_inputtohidden + B1)
print("\n This is the shape of y (input to hidden)\n", y.shape)
print("\n This is the length of y (input to hidden)\n", len(y))
print("\nThis is y (input to hidden):\n", y) 

# 2nd activation function (hidden to output)

inputtohidden_times_weights_hiddentooutput = torch.mm(y, W2)
print("\nThis is input to hidden mm weights (hidden to output):\n", inputtohidden_times_weights_hiddentooutput) 

y2 = activation(inputtohidden_times_weights_hiddentooutput + B2)
print("\n This is the shape of y (hidden to output)\n", y2.shape)
print("\n This is the length of y (hidden to output)\n", len(y2))
print("\nThis is y (hidden to output):\n", y2) 


def softmax(vector):
    e = exp(vector)
    return e / e.sum()

softmax_activation = softmax(y2)
print("\nThis is softmax activation:\n", softmax_activation)
