import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

#using r is good practice for string literals like file paths and stuff
#also using float32 halves the amount of memory consumption which is great bc we have images here
#so it doesnt really matter if we have 64bit
train = pd.read_csv(r"/Users/mokshith/Downloads/DigitRecognizer/train.csv", dtype = np.float32)

#all the data preprocessing and handling is the same as log regression

#splitting the data into features and labels
#also .values returns a numpy array containing all the data
targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != "label"].values/255

#splitting into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features_numpy, targets_numpy, test_size=0.2, random_state=42
)

#creating tensors
#btw tensors can be moved to a GPU, numpy arrays cant
featuresTrain = torch.tensor(features_train)
targetsTrain = torch.tensor(targets_train).type(torch.LongTensor)

#labels for classification tasks need to be LongTensor
#the target labels need to be in the form of integer indices
#this is what the models loss function expects as well
featuresTest = torch.tensor(features_test)
targetsTest = torch.tensor(targets_test).type(torch.LongTensor)

#setting batch size, iteration size, and epochs
batch_size = 100
n_iters = 10000
#mini-batches per epoch = (total samples)/ (batch size)
num_mini_batches = (len(features_train) / batch_size)
#number of epochs = (total iterations) / (mini-batches)
num_epochs = n_iters/ num_mini_batches
num_epochs = int(num_epochs)

#creating pytorch training and testing data
#this makes tuples
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

#creating dataloaders
#this allows us to iterate over the data in batches
#for each epoch, it provides a batch of input features and their target labels
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ArtificialNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        fc1_output = self.fc1(x)
        relu = self.relu1(fc1_output)
        fc2_output = self.fc2(relu)
        tanh = self.tanh2(fc2_output)
        fc3_output = self.fc3(tanh)
        elu = self.elu3(fc3_output)
        fc4_output = self.fc4(elu)

        return fc4_output

#defining our network
input_dim = 28*28 #pixels
hidden_dim = 150 #hyperparameter
output_dim = 10 #number of classes

model = ArtificialNeuralNetwork(input_dim, hidden_dim, output_dim)
error = nn.CrossEntropyLoss()

learning_rate = .01 #also a hyperparameter
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        train = images.view(-1,28*28)
        labels = labels

        #have to clear gradients
        optimizer.zero_grad()

        #results
        outputs = model(train)

        #loss
        loss = error(outputs, labels)

        #gradients with respect to each parameter
        loss.backward()

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
