import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#loading in data
train = pd.read_csv(r"/Users/mokshith/Downloads/DigitRecognizer/train.csv", dtype = np.float32)
#looking at the first 10 rows
print(train.head(10))

#train.label accesses the label column and .values converts this into a numpy array
#so this stores the target labels as a numpy array
targets_numpy = train.label.values

#.loc is an indexer to select rows or columns. here all rows are selected, and the label column is excluded
#.values turns all the feature columns into a numpy array
#then we divide by 255 to normalize our values
features_numpy = train.loc[:,train.columns != "label"].values/255

#splitting data into training and testing sets. 80% is training, set a random seed for reproducibility
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42)
#converting to tensors
featuresTrain = torch.tensor(features_train)
#need to set datatype because loss functions expect target labels to be of integer type
targetsTrain = torch.tensor(targets_train).type(torch.LongTensor)

featuresTest = torch.tensor(features_test)
targetsTest = torch.tensor(targets_test).type(torch.LongTensor)

#epoch is a complete pass through the entire training dataset
#iterations are the number of times models params are updated
#batch is a subset of training data and its used to compute gradient and update params
#good to train on mini-batches for efficiency and convergence speed
batch_size = 100
n_iters = 10000
#mini-batches per epoch = (total samples)/ (batch size)
num_mini_batches = (len(features_train) / batch_size)
#number of epochs = (total iterations) / (mini-batches)
num_epochs = n_iters/ num_mini_batches
num_epochs = int(num_epochs)

#creating tensor datasets
#this wraps data tensors
#creates dataset where each sample consists of a tuple of tensors (feature and target tensors here)
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

first_sample = train[0]
features_tensor = first_sample[0]

target_tensor = first_sample[1]

##visulizing these tensor datasets. i commented it out
"""
print("Features tensor:", features_tensor)
print("Target tensor:", target_tensor)
"""

#creating data loaders
#DataLoader provides an iterable over a given dataset. Supports batching, shuffling, etc.
#this loarder is used to iterate over the datasets in batches
#for each epoch it provides a batch of input features and their target labels
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

#visualizing an image in dataset. i commented out the lines of code
"""
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()
"""

#creating the actual logistic regression model
#it is a subclass of nn.Module as all other in PyTorch
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        #calling super constructor
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    #forward same as linear regression as log regression is essentially the same but with sigmoid function
    def forward(self, x):
        return self.linear(x)

    #btw the sigmoid function for log regression is handled within the loss function in PyTorch
    #it is applied internally (such as with Binary Cross Entropy with Logits)
    #this loss function takes in raw logits as input and applies the sigmoid function internally
    #this helps improve numerical stability and effiency

input_dim = 28*28 #size of the images
output_dim = 10 #number of classes

#initializing model
model = LogisticRegression(input_dim, output_dim)

#loss function for logistic regression is cross entrpy loss function
error = nn.CrossEntropyLoss()

#initializing optimizer
#btw model.parameters() collects all the parameters that the model has defined
#usually these parameters are associated w/ layers like nn.Linear
optimizer = torch.optim.SGD(model.parameters(), lr = .001)

#time to train the model!!!!!!
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    #train_loader loades data in batches from the training dataset 
    #each iteration over train_loader returns some batch of data 
    #using enumerate() here 
    #index value pairs. the value here is a tuple (images, labels) where images are the features 
    for i, (images, labels) in enumerate(train_loader):
        train = images.view(-1,28*28)
        labels = labels
        #zeroing out the gradients 
        optimizer.zero_grad()

        #forward propogation 
        outputs = model(train) 

        #getting the loss 
        loss = error(outputs, labels)

        #getting the gradients with respect to each parameter 
        loss.backward()

        #updating the params  
        optimizer.step()

        count += 1

        #making predictions 
        if count % 50 == 0: 
            correct = 0 
            total = 0 
            
            for images, labels in test_loader: 
                test = images.view(-1,28*28)

                outputs = model(test)
                
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            #storing loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            #printing loss 
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
