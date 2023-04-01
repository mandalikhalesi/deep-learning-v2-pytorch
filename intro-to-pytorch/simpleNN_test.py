import torch

def activation(x):
    #Sigmoid activation function
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)

#initialize features, weights, biases
features = torch.randn((1, 5))
print("\nThese are features: ", features) 

weights = torch.randn_like(features)
print("\nThese are weights: ", weights) 

features_times_weights = torch.sum(features*weights)
print("\nThis is features times weights: ", features_times_weights) 

features_mm_weights = torch.mm(features, weights.view(5, 1))
print("\nThis is features mm weights: ", features_mm_weights) 

bias = torch.randn((1, 1))
print("\nThis is bias: ", bias) 

y = activation(features_times_weights + bias)
print("\nThis is y (sum): ", y) 

y2 = activation(features_mm_weights + bias)
print("\nThis is y (mm): ", y2) 


