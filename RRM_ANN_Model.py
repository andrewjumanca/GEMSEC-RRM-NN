# -*- coding: utf-8 -*-
"""
A artificial, three-layered IP neural network designed to work with protein &
peptide EIIP values, convolution, and the PyTorch ML library in order to 
train a model that can accurately predict biomolecular binding affinity solely
based on the amino acid seequence of a biomolecular structure.

Created on Thu Aug 12 21:21:01 2021

@author: Andrew Jumanca
"""

import torch
import torch.nn as nn
import sys
import math
import pandas as pd


'''------------ Importing and Converting Data to Tensors ------------'''
from torch.utils.data import DataLoader

# Checking to see if data is already imported
if not all(var in globals() for var in ("train_eiip", "test1_eiip", "test2_eiip")):
    print("Importing Data")
    train_eiip = pd.read_csv('train_eiip.csv')
    test1_eiip = pd.read_csv('test1_eiip.csv')
    test2_eiip = pd.read_csv('test2_eiip.csv')
    
# Checking to see if all numerical values in sets are converted from strings
if isinstance(train_eiip['conv_length'][0], str):
    print("Parsing Data")
    import ast
    train_eiip.protein_EIIP=train_eiip.protein_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    train_eiip.peptide_EIIP=train_eiip.peptide_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    train_eiip.conv_length=train_eiip.conv_length.apply(lambda s: list(ast.literal_eval(s)))
    
    test1_eiip.protein_EIIP=test1_eiip.protein_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    test1_eiip.peptide_EIIP=test1_eiip.peptide_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    test1_eiip.conv_length=test1_eiip.conv_length.apply(lambda s: list(ast.literal_eval(s)))
    
    test2_eiip.protein_EIIP=test2_eiip.protein_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    test2_eiip.peptide_EIIP=test2_eiip.peptide_EIIP.apply(lambda s: list(ast.literal_eval(s)))
    test2_eiip.conv_length=test2_eiip.conv_length.apply(lambda s: list(ast.literal_eval(s)))

# Data and Labels pulled from dataframes
x_train = train_eiip['conv_length']
y_train = train_eiip['pIC50']

x_test1 = test1_eiip['conv_length']
y_test1 = test1_eiip['pIC50']

x_test2 = test2_eiip['conv_length']
y_test2 = test2_eiip['pIC50']

# Conversion to Tensor objects
x_train, y_train, x_test1, y_test1, x_test2, y_test2 = map(torch.tensor,
                                                           (x_train, y_train,
                                                           x_test1, y_test1,
                                                           x_test2, y_test2))

y_train = torch.transpose(y_train.unsqueeze(0), 0, 1)
y_test1 = torch.transpose(y_test1.unsqueeze(0), 0, 1)
y_test2 = torch.transpose(y_test2.unsqueeze(0), 0, 1)

train_loader = DataLoader((x_train,y_train), batch_size=100, shuffle=True)
test1_loader = DataLoader((x_test1,y_test1), batch_size=100, shuffle=False)
test2_loader = DataLoader((x_test2,y_test2), batch_size=100, shuffle=False)

'''------------ Math & Layers -------------'''
#i = 10 # binding affinity categories
#
#
#sigma = torch.rand(375, dtype = torch.float64).reshape(1, 375)
#print(sigma)
#
#e = torch.randint(-8, 9, (20, 375), dtype = torch.float64, requires_grad = False)
#
#E = torch.mm(e, torch.transpose(sigma, 0, 1))
#
#B = torch.rand(i, 20, dtype = torch.float64, requires_grad = False)
#
#U = torch.mm(B, E)
#print(U)
#
#P = (math.e**U)/(1 + math.e**U)
#
#output = P.mean()
#print(output)
#
#output.backward()
#
# print(sigma.grad) no grad cuz linear

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

class CustomLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        # epsilon weight matrix:
        weights = torch.Tensor(output_size, input_size)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(output_size)
        self.bias = nn.Parameter(bias)
        
        # weight & bias initializations
        nn.init.kaiming_normal_(self.weights, a=0, mode='fan_in')
        # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        # bias init
        nn.init.uniform_(self.bias, -bound, bound)
        
        def forward(self, sigma):
            epsilon_times_sigma = torch.mm(self.weights.t(), sigma)
            return torch.add(epsilon_times_sigma, self.bias) #e times x + b



'''------------ BLANK ----------------'''


class Model(nn.Module):
    def __init__(self, in_sz, out_sz, layers, bias=True):
        super().__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz

        layerlist = []
        
        layerlist.append(CustomLinearLayer(366, 20))
        layerlist.append(CustomLinearLayer(20, 10))
        layerlist.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layerlist)
        
    def forward(self, x):
        x, y = x.shape
        if y != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
        x = self.layers(x)
        return max(x)
        
model = Model(366, 10, [366,20])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# EPOCHS
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_correlation = 0
    test_correlation = 0
    
    # Training Batches:
    print("----------------------------- BREAK --------------------------")
    for j, (x_train, y_train) in enumerate(train_loader): #fix this line
        j+=1
        print("----------------------------- BREAK --------------------------")
        # Applying Model
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Counting Correct Predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_correlation = (predicted == y_train).sum()
        train_correlation += batch_correlation
        
        # Updating Parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print interim results
        if j%1 == 0:
            print(f'epoch: {i:2}  batch: {j:4} [{100*j:6}/60000]  loss: {loss.item():10.8f}  \
accuracy: {train_correlation.item()*100/(100*j):7.3f}%')
    
    # Update train loss & accuracy for the epoch
    train_losses.append(loss)
    train_correct.append(train_correlation)
    
    # Run the testing batches
    with torch.no_grad():
        for b, (x_test1, y_test1) in enumerate(test1_loader):

            # Apply the model
            y_val = model(x_test1)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            test_correlation += (predicted == y_test1).sum()
    
    # Update test loss & accuracy for the epoch
    loss = criterion(y_val, y_test1)
    test_losses.append(loss)
    test_correct.append(test_correlation)
    



