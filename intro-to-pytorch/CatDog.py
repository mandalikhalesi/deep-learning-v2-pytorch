# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import helper
import matplotlib.pyplot as plt
import time

# Define two transformers for training and testing
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomAutocontrast(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                    ])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()
                                    ])

# Access then load Cat/Dog training dataset
print("\n Accessing and loading local Cat/Dog training dataset...\n")
trainset = datasets.ImageFolder('~/Downloads/Cat_Dog_data/train/', transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
print("\n ...Access and load of Cat/Dog training dataset done.\n")

# Access then load Cat/Dog test dataset
print("\n Accessing and loading local Cat/Dog test dataset...\n")
testset = datasets.ImageFolder('~/Downloads/Cat_Dog_data/test/', transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
print("\n ...Access and load of Cat/Dog test dataset done.\n")

# Make loaded training dataset iterable
images, labels = next(iter(trainloader))

# Check if image files are displaying OK from the dataset - looking good
# helper.imshow(images[0], normalize=False)

# Build a feed-forward network with LogSoftmax as output, add two dropout operations after each ReLU activation
model = nn.Sequential(nn.Linear(150528, 128),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)
                      )

# Define a loss function using Negative Log Likelihood Loss function
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Start training, add epochs
print("\n Training NN from training data (2 hidden layers, 5 epochs, optim.Adam, lr 0.005)...\n")

epochs = 5
counter = 0

train_losses, test_losses, epoch_trainingtime = [], [], []

for e in range(epochs):
   tot_train_loss = 0
   training_time = 0
   start = time.time()

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

         #Turn off dropouts in the model during inference
         model.eval()

         for images_test, labels_test in testloader:
   
            # Flatten images 
            images_test = images_test.view(images_test.shape[0], -1)

            # Forward pass, get logits
            logits_test = model(images_test)

            # Calculate loss with logits and labels
            loss_test = criterion(logits_test, labels_test)

            tot_test_loss += loss_test.item()

         # Set model back to regular training mode
         model.train()
         end = time.time()

      # Get mean loss to enable comparison between training and test sets
      train_loss = tot_train_loss/len(trainloader.dataset)
      test_loss = tot_test_loss/len(testloader.dataset)
      training_time = (end - start)/3600
      #test_accuracy = test_correct/len(testloader.dataset)
      
      # At completion of epoch
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      epoch_trainingtime.append(training_time)
      
      counter = counter + 1
      
      print("\nEpoch {} of {}:".format(counter,epochs))
      print("\nTraining loss:\n{}".format(train_loss))
      print("\nTest loss:\n{}".format(test_loss))
      print("\nEpoch training time (h):\n{}".format(training_time))

# Save the model to external checkpoint file
print("\nCat/Dog model checkpoint saved to file...\n")
torch.save(model.state_dict(), 'checkpoint_CatDog_2HiddenLayers.pth')

print("\n...and we're done with calculations. Now to plot the losses and training times.\n")

# Plot results in interactive window
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.plot(epoch_trainingtime, label='Epoch training time (h)')
plt.title("Training vs Validation loss")
plt.legend(frameon=False)

# %%
