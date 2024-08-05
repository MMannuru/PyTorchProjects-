import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

#creating LinearRegression class (parent class is nn.module)
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        #calling super constructor to initilize internal state of inherited attributes 
        super(LinearRegression, self).__init__()
        #nn contains the building blocks for neural nets (layers, activation functions, and loss functions)
        #this applies linear transformation to the incoming data 
        self.linear = nn.Linear(input_size, output_size)

    #forward pass of the linear regression model 
    #x is the input tensor to the model 
    #self.linear is an instance of nn.Linear defined in constructor 
    def forward(self, x):
        return self.linear(x)

#below we have some very simple data to use for linear regression 
#we convert to numpy arrays then to tensors 
#this is input data 
car_prices_array = [3,4,5,6,7,8,9]
car_prices_array_np = np.array(car_prices_array, dtype = np.float32)
car_prices_array_np = car_prices_array_np.reshape(-1,1)
car_price_tensor = torch.tensor(car_prices_array_np)

#these are the actual values that we want to try to predict 
number_of_car_sell_array = [7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = torch.tensor(number_of_car_sell_np)


#we create an instance of LinearRegression class here and we specifiy input/output dimensions 
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

#mean sqaured error as loss function 
mse = nn.MSELoss()


learning_rate = 0.02
#initialize an optimizer here. using SGD (computationally efficent vs BGD)
#torch.optim is a module in pytorch with a bunch of optimization algs, theyre used to adjust model params 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []

iteration_number = 100
#training loop. 1000 iterations 
for iteration in range(iteration_number):
    #zero the gradients 
    #this step is so important! gradients accumulate by default 
    optimizer.zero_grad()

    #forward pass 
    results = model(car_price_tensor)
    #loss with mean sqaured error 
    loss = mse(results, number_of_car_sell_tensor)
    #getting gradients of each param with respect to loss function 
    loss.backward()
    #updating weights 
    optimizer.step()

    #adding to list 
    loss_list.append(loss.data)

    #tracking our results 
    if(iteration % 50 == 0):
        print("Iteration {}, loss {}".format(iteration, loss.data))


#making some predictions based on some data 
#.detach is really important
#creates a new tensor that isnt part of computation graph so we dont accumulate gradients 
predicted = model(car_price_tensor).detach().numpy()

#plotting results 
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data",color ="red")
plt.scatter(car_prices_array,predicted,label = "predicted data",color ="blue")

plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()
