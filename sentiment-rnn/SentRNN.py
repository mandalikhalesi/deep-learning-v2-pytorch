# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

## 1 - DATA PREP
# Open text file and read in data as 'text'
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()

# print("\nThe first 2000 chars from reviews:\n{}\n".format(reviews[:2000]))
# print("\nThe first 100 chars from labels:\n{}\n".format(labels[:100]))
# print("\nThis is string.punctuation():\n{}\n".format(punctuation))

# Remove punctuation from reviews
reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])
# print("\nThe first 2000 chars from reviews (no punct.):\n{}\n".format(all_text[:2000]))

# Data split by newline chats
reviews_split = all_text.split('\n')
# print("\nThe first 2000 chars from reviews_split:\n{}\n".format(all_text[:2000]))
all_text = ' '.join(reviews_split)
# print("\nThe first 2000 chars from reviews (no punct., newline):\n{}\n".format(all_text[:2000]))

# Create list of words from all_text
words = all_text.split()
# print("\nThe first 300 words (no punct., newline):\n{}\n".format(words[:300]))

## 2 - ENCODE DATA
# Encode words into integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
# print("\nUnique words: {}\n".format(len(vocab_to_int)))

# Use vocab_to_int dictionary to tokenize each review i reviews_int
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# print("\nThe first tokenized review from reviews_split:\n{}\n".format(reviews_ints[:1]))

labels_split = labels.split('\n')
enc_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
# print("\nThe first 300 labels from labels_split:\n{}\n".format(labels_split[:300]))
# print("\nThe first 300 labels(enc.) from enc_labels:\n{}\n".format(enc_labels[:300]))

## 3 - CLEAN DATA
# Review outlier stats
review_lens = Counter([len(x) for x in reviews_ints])
# print("\nZero-length reviews: {}\n".format(review_lens[0]))
# print("\nMax review length: {}\n".format(max(review_lens)))

# Remove outliers from data (ie len = 0)
# print("\nNo of reviews before removing outliers: {}\n".format(len(reviews_ints)))
nonzero_goodlength_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
reviews_ints = [reviews_ints[ii] for ii in nonzero_goodlength_idx]
enc_labels = np.array([enc_labels[ii] for ii in nonzero_goodlength_idx])
# print("\nNo of reviews after removing outliers: {}\n".format(len(reviews_ints)))

# Pad with zeros if necessary
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with zeros
        or truncated to the input length'''
    
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

seq_length = 300
features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print("\nFirst row of features dataset:\n{}\n".format(features[0]))
# print("\n25000th row of features dataset:\n{}\n".format(features[-1:]))

# print("\nFirst ten rows of features dataset:\n{}\n".format(features[:10]))
# print("\nLast 10 values of first 30 batches:\n{}\n".format(features[:30,-10:]))
# print("\n'Length' of features dataset: {}\n".format(len(features)))

## 4 - CREATE TRAINING, VALIDATION, TEST DATA
# Group data into training data (with ratio of split_frac), and remaining data into validation and test datasets (half and half)
split_frac = 0.8
valid_frac = (1-split_frac)/2

train_x = features[:int((len(features)*split_frac))]
valid_x = features[int(len(features)*split_frac):int((len(features)*(split_frac+valid_frac)))]
test_x = features[(int(len(features)*(split_frac+valid_frac))):]

train_y = enc_labels[:int((len(enc_labels)*split_frac))]
valid_y = enc_labels[int(len(enc_labels)*split_frac):int((len(enc_labels)*(split_frac+valid_frac)))]
test_y = enc_labels[(int(len(enc_labels)*(split_frac+valid_frac))):]

# print("\n'Shape' of train_x dataset: {}".format(train_x.shape))
# print("\n'Shape' of valid_x dataset: {}".format(valid_x.shape))
# print("\n'Shape' of test_x dataset: {}".format(test_x.shape))

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

## 5 - LOAD DATA
batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

sample_x, sample_y = next(iter(train_loader))

# print("\nSample input size: {}".format(sample_x.size()))
# print("\nSample input:\n{}\n".format(sample_x))
# print("\nSample label size: {}".format(sample_y.size()))
# print("\nSample label:\n{}\n".format(sample_y))

train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):
    print("\nTraining on GPU...\n")
else:
    print("\nNo GPU available, training on CPU...\n")

## 6 - DEFINE MODEL, LAYERS AND INITIALISE
class SentimentRNN(nn.Module):
    '''
    The RNN model that will be used for Sentiment analysis
    '''

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        '''
        Initialise the model by setting up the layers
        '''

        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Define layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        '''
        Forward pass of model
        '''
        
        batch_size = x.size(0)

        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.LSTM(embeds, hidden)
        lstm_out = lstm_out[:,-1,:] # getting the last timestep output only
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        sig_out = self.sig(out)

        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        '''
        Initialise hidden state (two tensors initialised to zero for hidden state and cell state of LSTM)
        '''

        weight = next(self.parameters()).data

        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                        weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                        weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
                        
        return hidden


## 7 - INSTANTIATE
# Instantiate the network

vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print("\nThis is the net:\n{}\n".format(net))

## 8 - TRAIN MODEL
# Train the model

lr = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# Training values

epochs = 4
counter = 0
print_every = 10
clip = 5 

if(train_on_gpu):
    net.cuda()


net.train()
print("\nTraining model...\n")

for e in range(epochs):
    h = net.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        start = time.time()

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
    
        h = tuple([each.data for each in h]) # Creating new variables for the hidden state

        net.zero_grad()

        # print("\nh.size (before): {}\n".format(len(h)))

        output, h = net(inputs, h)
        # print("\noutput.size: {}\n".format(output.size()))
        # print("\nh.size (after): {}\n".format(len(h)))
        # print("\ninputs.size: {}\n".format(inputs.size()))
        # print("\noutput(squeeze).size: {}\n".format(output.squeeze().size()))
        # print("\nlabels.size: {}\n".format(labels.size()))

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            
            for inputs, labels in valid_loader:
                
                val_h = tuple([each.data for each in val_h])
                
                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())
        
            net.train()
            end = time.time()

            print("\nEpoch: {}/{}; ".format(e+1, epochs))
            print("Step: {}; ".format(counter))
            print("Loss: {:.5f}; ".format(loss.item()))
            print("Val Loss: {:.5f}; ".format(np.mean(val_losses)))
            print("Time/step: {:.1f}(min);\n".format((end-start)/60))

print("\n...model training complete.\n")


# %%

