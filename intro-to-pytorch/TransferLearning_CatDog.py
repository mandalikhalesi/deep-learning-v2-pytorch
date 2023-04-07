# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.optim as optim
import six.moves.urllib as urllib #pyright: ignore (acc to Github discussion on why Pylance throws error)
import helper
import matplotlib.pyplot as plt
import time

# Adding hack from https://github.com/pytorch/pytorch/issues/33288 as getting SSL certificate verify failed error (unable to get local issuer certificate) when downloading DenseNet121 pretrained model (Mac issue?)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# urllib section as usual
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Define two transformers for training and testing, use normalization params expected by DenseNet121 pretrained model
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomAutocontrast(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                    ])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                    ])

# Access then load Cat/Dog training dataset
print("\n Accessing and loading local Cat/Dog training dataset (batch=64)...\n")
trainset = datasets.ImageFolder('~/Downloads/Cat_Dog_data/train/', transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
print("\n ...Access and load of Cat/Dog training dataset done.\n")

# Access then load Cat/Dog test dataset
print("\n Accessing and loading local Cat/Dog test dataset (batch=64)...\n")
testset = datasets.ImageFolder('~/Downloads/Cat_Dog_data/test/', transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
print("\n ...Access and load of Cat/Dog test dataset done.\n")

# Make loaded training dataset iterable
#images, labels = next(iter(trainloader))

# Print out model to see what's in the box - looking good
model = models.densenet121(pretrained = True)
print("This is the model architecture:\n{}".format(model))

for param in model.parameters():
   param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
   ('fc1', nn.Linear(1024, 500)),
   ('relu', nn.ReLU()),
   ('fc2', nn.Linear(500,2)),
   ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

# Setting cuda to False as not supported on Mac

for cuda in [False]:
   criterion = nn.NLLLoss()
   optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

   start_run = time.time()

   for ii, (inputs, labels) in enumerate(trainloader):
      inputs, labels = Variable(inputs), Variable(labels)
      #if cuda:
      #   inputs, labels = inputs.cuda(), labels.cuda()

      start = time.time()
      print("\nStart batch no.{}...\n".format(ii + 1))

      # Forward pass, loss function
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      end = time.time()
      print("...End batch no.{}. Time per batch: {:.3f} seconds.\n".format(ii + 1, (end - start)))

      if ii == 3:
         break
   
   end_run = time.time()

   print("\nAvg time per batch: {:.3f} seconds".format((end_run - start_run)/(ii + 1)))

# %%
