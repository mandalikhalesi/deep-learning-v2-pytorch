import torch

def activation(x):
    #Sigmoid activation function
    return 1 / (1 + torch.exp(-x))

# Features are 3 random normal variables
features = torch.randn((1, 3))
print("\nThese are features: ", features) 

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
print("\nThese are weights (input to hidden): ", W1) 
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)
print("\nThese are weights (hidden to output): ", W2) 

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
print("\nThese are biases (input to hidden): ", B1) 

B2 = torch.randn((1, n_output))
print("\nThese are biases (hidden to output): ", B2) 



#initialize features, weights, biases

#weights = torch.randn_like(features)
#print("\nThese are weights: ", weights) 

features_times_weights_inputtohidden = torch.mm(features, W1)
print("\nThis is features mm weights (input to hidden): ", features_times_weights_inputtohidden) 

#features_mm_weights = torch.mm(features, weights.view(5, 1))
#print("\nThis is features mm weights: ", features_mm_weights) 

#bias = torch.randn((1, 1))
#print("\nThis is bias: ", bias) 

y = activation(features_times_weights_inputtohidden + B1)
print("\nThis is y (input to hidden): ", y) 

inputtohidden_times_weights_hiddentooutput = torch.mm(y, W2)
print("\nThis is input to hidden mm weights (hidden to output): ", inputtohidden_times_weights_hiddentooutput) 

y2 = activation(inputtohidden_times_weights_hiddentooutput + B2)
print("\nThis is y (hidden to output): ", y2) 


