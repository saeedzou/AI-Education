# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 19:01:34 2021

NEURAL NETWORKS

@author: Dexter
"""
## nn depends on Autograd to define models and differentiate them
# Typical procedure to define neural networks
    # Define the neural network with some learnable parameters
    # Iterate over input
    # process input over the network
    # calculate the error(loss)
    # backprop

import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)

print(len(list(net.parameters())))
print(list(net.parameters())[0].size())

#  a random 32*32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

## Loss Function

output = net(input)
target = torch.rand(10).view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# gradients usually accumulate so we should zero it every iteration
net.zero_grad()
print('berfore')
print(net.conv1.bias.grad)
loss.backward()
print('after')
print(net.conv1.bias.grad)
# update

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()