#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:32:37 2017

@author: wty
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input size is 28x28
        #28 to 24
        self.conv1 = nn.Conv2d(1, 6, 5)
        #As described in the paper Efficient Object Localization Using Convolutional Networks ,
        #if adjacent pixels within feature maps are strongly correlated 
        #(as is normally the case in early convolution layers) 
        #then iid dropout will not regularize the activations 
        #and will otherwise just result in an effective learning rate decrease.
        self.conv_drop = nn.Dropout2d()
        #24 subsampling to 12, 12 to 8
        self.conv2 = nn.Conv2d(6, 16, 5)
        #8 subsampling to 4, 16 chanels
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(self.conv_drop(out), 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        #The input size is N * 10, every slice along second dimension will sum to 1
        out = F.log_softmax(out,dim=1)
        return out
    
    def predict(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(self.conv_drop(out), 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        #softmax output the raw probability
        out = F.softmax(out,dim=1)
        
        return out
    
    