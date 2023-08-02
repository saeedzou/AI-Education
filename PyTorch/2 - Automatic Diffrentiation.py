# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:11:56 2021

AUTOMATIC DIFFERENTIATION

@author: Dexter
"""

# Pytorch AutoGrad
import torch, torchvision
from torch import nn

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
# Forward pass
prediction = model(data)
# Calculate loss and backprop
loss = (prediction - labels).sum()
loss.backward()

# Optimizer. calling .step() adjusts each model parameters by its gradients
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optimizer.step()

## Differentiation in Autograd

a = torch.tensor([4., 2.5], requires_grad=True)
b = torch.tensor([2.15, 3.5], requires_grad=True)
Q = 4*a**5 - 2*b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(20*a**4 == a.grad, -4*b == b.grad)
# torch.autograd tracks operations on all tensors which have their
# requires_grad flag set to True. For tensors that don’t require gradients, 
# setting this attribute to False excludes it from the gradient computation DAG.

# Sometimes we need to freeze part of the model parameters for benefiting the computation
# and other reasons and only adjusting the parameters in the last layers like
# finetuning a pretrained resnet

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Linear(512, 10)
# Only the parameters of the model.fc will be adjusted in backprop