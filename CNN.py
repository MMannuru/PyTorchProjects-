import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

#once again same data preprocessing as before
#using dtype=np.float32 to half memory consumption
#also 32bit is the standard for most ML tasks anyway
train = pd.read_csv(r"/Users/mokshith/Downloads/DigitRecognizer/train.csv", dtype = np.float32)

#splitting data into targets and features
targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != "labels"].values/255

#splitting into training sets and testing sets
features_Train, features_Test, targets_Train, targets_Test = train_test_split(
    features_numpy, targets_numpy, train_size=0.8, random_state=42)

#creating tensors
#tensors can be moved to a GPU, numpy arrows cannot
#labels for classification tasks need to be LongTensor
#the target labels need to be in the form of integer indices
#this is what the models loss function expects as well
featuresTrain = torch.tensor(features_Train)
targetsTrain = torch.tensor(targets_Train).type(torch.LongTensor)

featuresTest = torch.tensor(features_Test)
targetsTest = torch.tensor(targets_Test).type(torch.LongTensor)

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        #first convolution
        #in_channels = 1 because we are dealing with grey scale channels
        #out_channels is the number of filters we want to apply, here it is 16
        #each filter learns to detect a different feature
        #each one has a different feature map
        #kernel_size = 5 for a 5x5 filter
        #stride = 2 means filter moves 2 pixels
        #padding = 0 means no padding, filter only applies where it fits
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 5, stride = 2, padding = 0)
        #relu activation function!!! this is for non-linearity so model can learn complex patterns
        self.relu1 = nn.ReLU()
        #pooling! we use maxpooling to downsample the image
        #we use 2x2 window here (max value from each window is taken)
        #this reduces width and height by a factor of 2
        #this effictively retains all the important from original image but reduces dimensions greatly
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        #2nd convolution
        #the in_channels in this layer are the number of filters from the previous layer
        #here out_channels is 32 so we have 32 filters
        #again 5x5 kernel here
        #stride = 1 so we move one pixel
        #no padding again
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        #relu activation function again
        self.relu2 = nn.ReLU()
        #max pooling to reduce dimensions
        #2x2 window so both height and width are reduced by a factor of 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        #fully connected layer
        #we flatten it out so we can acc feed it into the layer
        #once again softmax is handeled by the cross-entropy loss function
        #output size is the number of classes which is 10
        #each element of this vector corresponds with a logit, and we feed this into softmax to
        #get predictions
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

        #overall each layer extracts more complex features, reduces dimensions, and then we classify based on this input

    def forward(self, x):
        #first convolution
        #we pass in the original input to the first conv layer
        output = self.cnn1(x)
        #send this through activation function
        output = self.relu1(output)
        #maxpooling
        first_conv_output = self.maxpool1(output)

        #2nd convolution layer
        output_two = self.cnn2(first_conv_output)
        #activation function
        output_two = self.relu2(output_two)
        #maxpooling
        second_conv_output = self.maxpool2(output_two)

        #fully connected layer
        #flattening image
        final_output = second_conv_output.view(second_conv_output.size(0), -1)
        final_output = self.fc1(second_conv_output)

        return final_output

#setting batch size, iteration size, and epochs
batch_size = 100
n_iters = 10000
#mini-batches per epoch = (total samples)/ (batch size)
num_mini_batches = (len(features_Train) / batch_size)
#number of epochs = (total iterations) / (mini-batches)
num_epochs = n_iters/ num_mini_batches
num_epochs = int(num_epochs)

#creating pytorch training and testing sets (these are tuples)
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)


#creating dataloaders
#this allows us to iterate over the data in batches
#for each epoch, it provides a batch of input features and their target labels
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

model = ConvolutionalNeuralNetwork()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #learning rate hyperparmeter that i experimented with


count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        train = images.view(100,1,28,28)
        labels = labels

        #have to clear gradients
        optimizer.zero_grad()

        #results
        outputs = model(train)

        #loss
        loss_value = loss(outputs, labels)
        #gradients with respect to each parameter
        loss_value.backward()

        optimizer.step()

        count += 1

        #making predictions
        if count % 50 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                test = images.view(100,1,28,28)

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
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss_value.data, accuracy))

