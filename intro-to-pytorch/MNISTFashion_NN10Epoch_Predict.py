# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import helper

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
print("\n Downloading and loading for MNIST Fashion dataset...\n")
trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print("\n ...Download and load for MNIST Fashion done.\n")

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
optimizer = optim.Adam(model.parameters(), lr=0.005)

print("\n Training NN (2 hidden layers, optim.Adam, lr 0.005)...\n")
## Add epochs
epochs = 3
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

print("\n...NN Training done.\n")

# Get data from iterable dataiter 
print("\n Checking predictions for MNIST Fashion...\n")

images, labels = next(dataiter)

img = images[0].view(1, 784)

# Turn off gradients for inference step

with torch.no_grad():
   logits2 = model.forward(img)

# Output of network are logits2, need to take softmax for probabilities
ps = F.softmax(logits2, dim=1)

print("\n...PS is:\n", ps)
print("\n...exp(PS) is:\n", torch.exp(ps))

helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')

print("\n...Predictions complete\n")

# %%
