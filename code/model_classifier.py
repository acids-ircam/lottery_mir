#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:41:02 2020

@author: esling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import LotteryClassification

"""

Here we define a dilated 1d-CNN model for general purpose waveform classification
This model will serve for both the tasks of
- Instrument recongition
- Singing mode classification

"""

# Global variables
instruments = ['bn','cl','db','fl','hn','ob','sax','tba','tbn','tpt','va','vc','vn']

class WaveformCNN(nn.Module):
    
    def __init__(self, args):
        super(WaveformCNN, self).__init__()
        in_size = np.prod(args.input_size)
        out_size = np.prod(args.output_size)
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        channels = args.channels
        n_mlp = args.n_layers
        # Create modules
        modules = nn.Sequential()
        self.in_size = in_size
        size = in_size
        in_channel = 1 
        kernel = args.kernel
        stride = kernel // 16
        """ First do a CNN """
        for l in range(n_layers):
            dil = ((args.dilation == 3) and (2 ** l) or args.dilation)
            pad = 3 * (dil + 1)
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, nn.Conv1d(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm1d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.25))
            size = int((size+2*pad-(dil*(kernel-1)+1))/stride+1)
        modules[-1].unprunable = True
        self.net = modules
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size) or hidden_size
            out_s = (l == n_mlp - 1) and out_size or hidden_size
            self.mlp.add_module('h%i'%l, nn.Linear(in_s, out_s))
            if (l < n_layers - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.mlp[-1].unprunable = True
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, x):
        x = x.view(-1, 1, self.in_size)
        out = x
        for m in range(len(self.net)):
            out = self.net[m](out)
        #print(out.shape)
        out = out.view(x.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        #print(out.shape)
        return out
    
        

"""

Model bottle inheritance

"""
class LotteryClassifierCNN(LotteryClassification, WaveformCNN):
    
    def __init__(self, args):
        super(LotteryClassifierCNN, self).__init__(args)
        WaveformCNN.__init__(self, args)
        self.pruning = args.pruning