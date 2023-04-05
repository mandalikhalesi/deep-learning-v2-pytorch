# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import helper
import matplotlib.pyplot as plt

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
print("\n Downloading and loading MNIST Fashion dataset...\n")
trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print("\n ...Download and load of MNIST Fashion dataset done.\n")

# Download then load test dataset
print("\n Downloading and loading test dataset...\n")
testset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
print("\n ...Download and load of test dataset done.\n")

# Build a feed-forward network with LogSoftmax as output
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
                      )

# Define a loss function using Negative Log Likelihood Loss function
#criterion = nn.NLLLoss(reduction=sum)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

images, labels = next(iter(trainloader))

# Start training, add epochs
print("\n Training NN from training data (2 hidden layers, optim.Adam, lr 0.005)...\n")

epochs = 15
counter = 0

train_losses, test_losses = [], []

for e in range(epochs):
   tot_train_loss = 0
  
   for images, labels in trainloader:
      # Clear gradients ahead of forward pass, backward pass, weights update
      optimizer.zero_grad()
   
      # Flatten images 
      images = images.view(images.shape[0], -1)

      # Forward pass, get logits
      logits_train = model(images)

      # Calculate loss with logits and labels
      loss_train = criterion(logits_train, labels)

      # Gradient and backward pass
      loss_train.backward()

      # Take an update step and view new weights
      optimizer.step()

      tot_train_loss += loss_train.item()

   else:
      # Get data from iterable testloader 
      images_test, labels_test = next(iter(testloader))
      tot_test_loss = 0
      test_correct = 0

      # Turn off gradients for inference step
      with torch.no_grad():
         for images_test, labels_test in testloader:
   
            # Flatten images 
            images_test = images_test.view(images_test.shape[0], -1)

            # Forward pass, get logits
            logits_test = model(images_test)

            # Calculate loss with logits and labels
            loss_test = criterion(logits_test, labels_test)

            tot_test_loss += loss_test.item()

            # Commenting out this as tensor sizes don't match... review test accuracy later
            #ps = torch.exp(logits_test)
            #top_p, top_class = ps.topk(1, dim=1)
            #equals = top_class == labels.view(*top_class.shape)
            #test_correct += equals.sum().item()

      # Get mean loss to enable comparison between training and test sets
      train_loss = tot_train_loss/len(trainloader.dataset)
      test_loss = tot_test_loss/len(testloader.dataset)
      #test_accuracy = test_correct/len(testloader.dataset)
      
      # At completion of epoch
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      
      counter = counter + 1
      
      print("\nEpoch {} of {}:\n".format(counter,epochs))
      print("\nTraining loss:\n{}".format(train_loss))
      print("\nTest loss:\n{}".format(test_loss))
      #print("\nTest accuracy:\n{}".format(test_accuracy))


print("\n...and we're done with calculations. Now to plot the losses.\n")

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.title("Training vs Validation loss")
plt.legend(frameon=False)

# %%
