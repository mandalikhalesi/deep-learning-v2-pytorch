import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import helper
import matplotlib_inline

# The MNIST datasets are hosted on yann.lecun.com that has moved under CloudFlare protection
# Run this script to enable the datasets download
# Reference: https://github.com/pytorch/vision/issues/1938

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Define a transformer to normalise data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download then load MNIST training dataset
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Make trainloader iterable
dataiter = iter(trainloader)

# Build a feed-forward network with LogSoftmax as output
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
                      )

# Define a loss function using Negative Log Likelihood Loss function
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

## Add epochs
epochs = 10
counter = 0
for e in range(epochs):
   running_loss = 0
   
   for images, labels in trainloader:
      # Clear gradients ahead of forward pass, backward pass, weights update
      optimizer.zero_grad()
   
      # Flatten images 
      images = images.view(images.shape[0], -1)

      # Forward pass, get logits
      logits = model(images)

      # Calculate loss with logits and labels
      loss = criterion(logits, labels)

      # Gradient and backward pass
      loss.backward()

      # Take an update step and view new weights
      optimizer.step()

      running_loss += loss.item()

   else:
      counter = counter + 1
      print("\nTraining loss epoch{}:\n{}".format(counter,running_loss/len(trainloader)))



# Get data from iterable dataiter 
print("\n Checking predictions...\n")

images, labels = next(dataiter)

img = images[0].view(1, 784)

with torch.no_grad():
   logits2 = model.forward(img)

# Output of network are logits2, need to take softmax for probabilities
ps = F.softmax(logits2, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)


# (comment out for for loop)
# Get data from iterable dataiter 
#images, labels = next(dataiter)
#print("\nType of images is:\n", type(images))
#print("\nimages.shape is:\n", images.shape)
#print("\nlabels.shape is:\n", labels.shape)

# Flatten images
#images = images.view(images.shape[0], -1)
#print("\nFlattened images:\n", images)
#print("\nFlattened images.shape is:\n", images.shape)

# Clear gradients ahead of forward pass, backward pass, weights update
#optimizer.zero_grad()

# Forward pass, get logits
#print ("\nStarting forward pass...\n")
#logits = model(images)
#print ("\nResulting logits are:\n", logits)

# Calculate loss with logits and labels
#loss = criterion(logits, labels)
#print ("\nResulting loss is:\n", loss)

#print ("\nGradient - before backward pass:\n", model[0].weight.grad)
#loss.backward()
#print ("\nGradient - after backward pass:\n", model[0].weight.grad)

# Take an update step and view new weights
#optimizer.step()
#print ("\nUpdated weights -\n", model[0].weight)

# (comment out for for loop)

