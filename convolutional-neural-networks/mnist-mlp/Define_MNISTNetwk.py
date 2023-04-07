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
import numpy as np

# Define two transformers for training and testing - just do simple version for now
train_transform = transforms.ToTensor()
#train_transform = transforms.Compose([transforms.RandomRotation(30),
#                                      transforms.RandomResizedCrop(224),
#                                      transforms.RandomAutocontrast(),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor()
#                                    ])

test_transform = transforms.ToTensor()
#test_transform = transforms.Compose([transforms.Resize(255),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor()
#                                    ])

# Define Batch size variable
batch_size = 20

# Access then load online MNIST training dataset
print("\n Accessing and loading MNIST training dataset...")
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
print("\n ...Access and load of MNIST training dataset done.\n")

# Access then load MNIST test dataset
print("\n Accessing and loading MNIST test dataset...")
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
print("\n ...Access and load of MNIST test dataset done.")

#Create another loader for accuracy testing usig same testset
accuracyloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
print("\n ...Load of MNIST accuracy dataset done.\n")

# Make loaded training dataset iterable
images, labels = next(iter(trainloader))

# Check if image files are displaying OK from the dataset
print("\nType of images is:\n", type(images))
print("\nImages.shape is:\n", images.shape)
print("\nLabels.shape is:\n", labels.shape)

# Define Model variables
inputpixelsize = 28 * 28 
hidden_layer1 = 512
hidden_layer2 = 512
dropout_rate = 0.2
learning_rate = 0.01

# Build a feed-forward network with LogSoftmax as output, add dropout operations after each ReLU activation
model = nn.Sequential(nn.Linear(inputpixelsize, hidden_layer1),
                      nn.ReLU(),
                      nn.Dropout(p=dropout_rate),
                      nn.Linear(hidden_layer1, hidden_layer2),
                      nn.ReLU(),
                      nn.Dropout(p=dropout_rate),
                      nn.Linear(hidden_layer2, 10),
                      nn.LogSoftmax(dim=1)
                      )

print("\nThis is the model:\n{}".format(model))

# Define a loss function using Cross Entropy Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Start training, add epochs
print("\nTraining NN from training data (2 hidden layers, 10 epochs, optim.SGD, lr 0.01)...\n")

epochs = 3
counter = 0
test_loss_min = np.Inf

train_losses, test_losses, epoch_trainingtime = [], [], []

for e in range(epochs):
   tot_train_loss = 0
   training_time = 0
   start = time.time()

   for images, labels in trainloader:
      # Clear gradients ahead of forward pass, backward pass, weights update
      optimizer.zero_grad()
   
      # Flatten images 
      images = images.view(-1,inputpixelsize)

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

      #counter = counter + 1
      
      print("\nEpoch {} of {}:".format(e+1,epochs))
      print("\nTraining loss: {:.5f}; ".format(train_loss))
      print("Test loss: {:.5f}; ".format(test_loss))
      print("Epoch training time (h):{:.5f}\n".format(training_time))

      # save model if test loss has decreased
      if test_loss <= test_loss_min:
         print('Test loss decreased ({:.5f} --> {:.5f}).  Saving model ...'.format(
         test_loss_min,
         test_loss))
         torch.save(model.state_dict(), 'testlossmin_model.pt')
         test_loss_min = test_loss

print("\n...and we're done with model training/testing. Now to plot the losses and training times.\n")

# Plot results in interactive window
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.plot(epoch_trainingtime, label='Epoch training time (h)')
plt.title("Training vs Validation loss")
plt.legend(frameon=False)

# # === - commenting this out for now as getting AttributeError: 'Tensor' object has no attribute 'images_accuracy'
# # Now test for accuracy
# print("\nLoading up saved test_loss_min model for accuracy testing...")
# model.load_state_dict(torch.load('testlossmin_model.pt'))
# print("\nThis is the model:\n{}".format(model))

# # initialize lists to monitor test loss and accuracy
# test_loss = 0.0
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# images_accuracy, labels_accuracy = next(iter(accuracyloader))

# # Turn off gradients for inference step
# with torch.no_grad():

#    model.eval() # prep model for evaluation

#    for images_accuracy, labels_accuracy in accuracyloader:
#        # Flatten images 
#        images_accuracy = images_test.view(images_accuracy.shape[0], -1)
#        # forward pass: compute predicted outputs by passing inputs to the model
#        output = model(images_accuracy)
#        # calculate the loss
#        loss = criterion(output, labels_accuracy)
#        # update test loss 
#        test_loss += loss.item()*images_accuracy.size(0)
#        # convert output probabilities to predicted class
#        _, pred = torch.max(output, 1)
#        # compare predictions to true label
#        correct = np.squeeze(pred.eq(labels_accuracy.images_accuracy.view_as(pred)))
      
#        # calculate test accuracy for each object class
#        for i in range(batch_size):
#           label = labels_accuracy.images_accuracy[i]
#           class_correct[labels_accuracy] += correct[i].item()
#           class_total[labels_accuracy] += 1

#    # calculate and print avg test loss
#    test_loss = test_loss/len(accuracyloader.dataset)
#    print('\nTest Loss: {:.5f}\n'.format(test_loss))

#    for i in range(10):
#       if class_total[i] > 0:
#          print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#          str(i), 100 * class_correct[i] / class_total[i],
#          np.sum(class_correct[i]), np.sum(class_total[i])))
#       else:
#          print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

# print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
#     100. * np.sum(class_correct) / np.sum(class_total),
#     np.sum(class_correct), np.sum(class_total)))

# %%
