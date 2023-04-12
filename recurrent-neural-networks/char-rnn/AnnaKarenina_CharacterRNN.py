# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Adding hack from https://github.com/pytorch/pytorch/issues/33288 as getting SSL certificate verify failed error (unable to get local issuer certificate) when downloading DenseNet121 pretrained model (Mac issue?)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# urllib section as usual
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Open text file and read in data as 'text'
with open('data/anna.txt', 'r') as f:
    text = f.read()

print("\nThe first 300 chars from Anna K:\n{}\n".format(text[:300]))

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])

print("\nThe first 300 chars (encoded) from Anna K:\n{}\n".format(encoded[:300]))

# Char RNN is expecting a one-hot encoded input -> create fuction to feed in text input as such
def one_hot_encode(arr, n_labels):
    
    #Initialize the array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1

    # Reshape to original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

# Getting familiar with the data
test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)
print("\nOne hot encode result for test_seq:\n{}\n".format(one_hot))

string_annak_300 = text[:300]
listchars = [*string_annak_300]
print("\nFirst 300 chars split up as chars in list:\n{}\n".format(listchars))
print("\nLength of list is:\n{} chars\n".format(len(listchars)))
print("\nLength of text is:\n{} chars\n".format(len(text)))

# Seeing how to reduce text to batchable size
def get_batches(arr, batch_size, seq_length):
    # getting a feel for this with actual data
    #batch_size = 10
    #seq_length = 10
    #q = len(text) // (batch_size * seq_length)
    #mod = len(text) % (batch_size * seq_length)
    #print("\nQuotient of text divided by seq_length of 10 and batch_size of 10:\n{}\n".format(q))
    #print("\nModulus of text divided by seq_length of 10 and batch_size of 10:\n{}\n".format(mod))
    q = len(arr) // (batch_size * seq_length)
    mod = len(arr) % (batch_size * seq_length)
    #n_batches = q

    #Slice data so that we have fully packed sequences (cut off some end text)
    arr = arr[:(len(arr)-mod)]
    arr = arr.reshape((batch_size, -1))

    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:+seq_length]
        # The targets shifted by one (why?)
        y = np.zeros_like(x)
        # This part below flying above my head...
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

# Check data with encoded chars from AnnaK text (first 10 items in sequence)
batches = get_batches(encoded, 8, 50)
x , y = next(batches)

print("\nx is:\n{}".format(x[:10, :10]))
print("\ny is:\n{}".format(y[:10, :10]))


# %%









# # Define two transformers for training and testing
# train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.ToTensor()
#                                     ])

# test_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.ToTensor()
#                                     ])

# # Define directory for Flower images, batch size and image classes 
# data_dir = './flower_photos/'
# batch_size = 20
# classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# # Access then load Flower training dataset
# start = time.time()
# print("\n Accessing and loading Flower training dataset...\n")
# train_dir = os.path.join(data_dir, 'train/')
# trainset = datasets.ImageFolder(train_dir, transform=train_transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# print("\n ...Access and load of training dataset done.\n")

# # Access then load Flower test dataset
# print("\n Accessing and loading Flower test dataset...\n")
# test_dir = os.path.join(data_dir, 'test/')
# testset = datasets.ImageFolder(test_dir, transform=test_transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
# print("\n ...Access and load of test dataset done.\n")

# # Make loaded training dataset iterable
# images, labels = next(iter(trainloader))
# images = images.numpy() # convert images to numpy for display

# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
#     plt.imshow(np.transpose(images[idx], (1, 2, 0)))
#     ax.set_title(classes[labels[idx]])

# # Print out some data stats
# print("\nNum training images: {}\n".format(len(trainset)))
# print("\nNum test images: {}\n".format(len(testset)))
# end = time.time()
# print("\n Time elapsed: {:.3f} seconds.\n".format(end - start))

# # Download pretrained model VGG16
# model = models.vgg16(pretrained=True)
# print("\nThis is the pretrained VGG16 model: {}\n".format(model))

# print(model.classifier[6].in_features) 
# print(model.classifier[6].out_features) 

# # Freeze training for all "features" layers
# for param in model.features.parameters():
#     param.requires_grad = False

# n_inputs = model.classifier[6].in_features
# print("\nThese are the in_features:\n {}\n".format(n_inputs))

# # add last linear layer (n_inputs -> 5 flower classes)
# # new layers automatically have requires_grad = True
# last_layer = nn.Linear(n_inputs, len(classes))

# model.classifier[6] = last_layer

# # check to see that your last layer produces the expected number of outputs
# print("\nThese are the last layer's out_features:\n {}\n".format(last_layer.out_features))

# # %%

# # Already have a model (VGG16), no need to build
# # model = nn.Sequential(nn.Linear(150528, 128),
# #                       nn.ReLU(),
# #                       nn.Dropout(p=0.2),
# #                       nn.Linear(128, 64),
# #                       nn.ReLU(),
# #                       nn.Dropout(p=0.2),
# #                       nn.Linear(64, 10),
# #                       nn.LogSoftmax(dim=1)
# #                       )

# # Define a loss function using Cross Entropy Loss and SGD function
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.classifier.parameters(), lr=0.001)

# # Start training, add epochs
# print("\n Training VGG16 model on training data (2 epochs, optim.SGD, lr 0.001)...\n")

# epochs = 2
# counter = 0

# train_losses, test_losses, epoch_trainingtime = [], [], []

# for e in range(epochs):
#    tot_train_loss = 0
#    training_time = 0
#    start = time.time()

#    for batch_i, (images, labels) in enumerate(trainloader):
#       # Clear gradients ahead of forward pass, backward pass, weights update
#       optimizer.zero_grad()

#       # Forward pass, get output
#       output_train = model(images)

#       # Calculate loss with output and labels
#       loss_train = criterion(output_train, labels)

#       # Gradient and backward pass
#       loss_train.backward()

#       # Take an update step and view new weights
#       optimizer.step()

#       tot_train_loss += loss_train.item()

#       if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
#          print('Epoch %d, Batch %d, Tot_train_loss: %.16f' %
#                (e+1, batch_i + 1, tot_train_loss / 20))
#          tot_train_loss = 0.0

#    else:
#       # Get data from iterable testloader 
#       images_test, labels_test = next(iter(testloader))
#       tot_test_loss = 0
#       class_correct = list(0. for i in range(5))
#       class_total = list(0. for i in range(5))

#       # Turn off gradients for inference step
#       with torch.no_grad():

#          #Turn off dropouts in the model during inference
#          model.eval()

#          for images_test, labels_test in testloader:

#             # Forward pass, get logits
#             output_test = model(images_test)

#             # Calculate loss with output and labels
#             loss_test = criterion(output_test, labels_test)

#             #tot_test_loss += loss_test.item()
#             tot_test_loss += loss_test.item()*images_test.size(0)

#             # convert output probabilities to predicted class
#             _, pred = torch.max(output_test, 1)    

#             # compare predictions to true label
#             correct_tensor = pred.eq(labels_test.data.view_as(pred))
#             correct = np.squeeze(correct_tensor.numpy())
#             np.squeeze(correct_tensor.cpu().numpy())
            
#             # calculate test accuracy for each object class
#             for i in range(batch_size):
#                label = labels_test.data[i]
#                class_correct[label] += correct[i].item()
#                class_total[label] += 1
            
#          # Set model back to regular training mode
#          model.train()
#          end = time.time()

#       # Get mean loss to enable comparison between training and test sets
#       train_loss = tot_train_loss/len(trainloader.dataset)
#       test_loss = tot_test_loss/len(testloader.dataset)
#       training_time = (end - start)/3600
#       #test_accuracy = test_correct/len(testloader.dataset)
      
#       # At completion of epoch
#       train_losses.append(train_loss)
#       test_losses.append(test_loss)
#       epoch_trainingtime.append(training_time)
      
#       counter = counter + 1
      
#       print("\nEpoch {} of {}:".format(counter,epochs))
#       print("\nTraining loss:\n{:.6f}".format(train_loss))
#       print("\nTest loss:\n{:.6f}".format(test_loss))
#       print("\nEpoch training time (h):\n{:.6f}\n".format(training_time))

#       for i in range(5):
#          if class_total[i] > 0:
#             print('\nTest Accuracy of %5s: %2d%% (%2d/%2d)' % (
#                classes[i], 100 * class_correct[i] / class_total[i],
#                np.sum(class_correct[i]), np.sum(class_total[i])))
#          else:
#             print('\nTest Accuracy of %5s: N/A (no training examples)' % (classes[i]))

#       print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)\n' % (
#          100. * np.sum(class_correct) / np.sum(class_total),
#          np.sum(class_correct), np.sum(class_total)))

# print("\n...and we're done with calculations. Now to plot the losses and training times.\n")

# # Plot results in interactive window
# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Validation loss')
# plt.plot(epoch_trainingtime, label='Epoch training time (h)')
# plt.title("Training vs Validation loss")
# plt.legend(frameon=False)

# %%
