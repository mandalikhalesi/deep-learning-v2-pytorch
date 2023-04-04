import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs (784) to hidden layer 1 (128) linear transformation
        self.hidden1 = nn.Linear(784, 128)
        # Hidden layer 1 inputs (128) to hidden layer 2 (64) linear transformation
        self.hidden2 = nn.Linear(128, 64)
        # Output layer (64) to 10 units - one for each digit
        self.output = nn.Linear(64, 10)

        # Define sigmoid activation function and softmax output (these lines not needed if expressed/wrapped using F)
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each operation
        # x = self.hidden(x)
        # x = self.output(x)
        # x = self.sigmoid(x)
        # x = self.softmax(x)

        # Alternatively (as F allows wrapping of self.sigmoid and self.softmax activations within one function):
        # Inputs to hidden layer 1 function with reLU activation
        x = F.relu(self.hidden1(x))
        # Hidden layer 1 inputs to hidden layer 2 function with reLU activation
        x = F.relu(self.hidden2(x))
        # Output layer with Softmax activation
        x = F.softmax(self.output(x), dim=1)
        
        return x
    

model = Network()
model
print ("\nThis is x (784 inputs to 128-relu to 64-relu to 10-softmax outputs):\n", model)