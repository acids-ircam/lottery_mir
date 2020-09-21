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
import models.sparsemax as sparsemax
from model import LotteryTranscription

"""

The following models are defined for various tasks of detection
- Onset detection (from spectral information)
- Drum transcription

"""

class MultiplySparsemax(nn.Module):
    """Multiplication-based sparsemax layers.
    It compute sparsemax over time and channel, separately, then multiply their outputs

    Args:
        sparsemax_lst (int): the 'frame' length of sparsemax over time.
    """

    def __init__(self, sparsemax_lst=64):
        super(MultiplySparsemax, self).__init__()
        self.lst = sparsemax_lst
        self.sparsemax_inst = sparsemax.Sparsemax(dim=-1)  # along insts
        self.sparsemax_time = sparsemax.Sparsemax(dim=-1)  # along time
        print('| and sparsemax_lst=%d samples at the same, at=`r` level' % self.lst)

    def forward(self, midis_out):
        """midis_ou: (batch, n_insts, time)"""
        batch, _, n_insts, time = midis_out.shape
        lst = self.lst
        len_pad = (lst - time % lst) % lst
        midis_out = F.pad(midis_out, [0, len_pad])
        midis_out_inst = self.sparsemax_inst(midis_out.transpose(2, 3)).transpose(2, 3)  # inst-axis Sparsemax
        midis_out_time = midis_out.reshape(batch, 2, n_insts, (time + len_pad) // lst, lst)
        midis_out_time = self.sparsemax_time(midis_out_time)
        midis_out_time = midis_out_time.reshape(batch, 2, n_insts, (time + len_pad))
        midis_final = midis_out_inst[:, :, :, :time] * midis_out_time[:, :, :, :time]
        return midis_final

class SpectralTranscriptionCNN(nn.Module):
    """
    State-of-art onset detection (and drum transcription) CNN based on
    Jan Schluter and Sebastian Bock, “Improved musical onset detection with convolutional neural networks,” 
    Proceedings IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2014.
    
    Instead of three independent networks, we only specify 
    - Single base processing layers
    - One specific layer for each output
    """
    
    def __init__(self, args):
        super(SpectralTranscriptionCNN, self).__init__()
        in_size = args.input_size
        out_size = args.output_size
        hidden_size = args.n_hidden
        n_layers = args.n_layers
        channels = args.channels
        n_mlp = args.n_layers
        self.in_size = in_size
        self.out_size = out_size
        # Create modules
        modules = nn.Sequential()
        size = in_size.copy()
        in_channel = 1 
        kernel = 5 #args.kernel
        stride = 1
        """ First do a CNN """
        for l in range(n_layers):
            dil = 1
            pad = kernel // 2
            in_s = (l==0) and in_channel or channels
            out_s = (l == n_layers - 1) and 1 or channels
            modules.add_module('c2%i'%l, nn.Conv2d(in_s, out_s, kernel, stride, pad, dilation = dil))
            if (l < n_layers - 1):
                modules.add_module('b2%i'%l, nn.BatchNorm2d(out_s))
                modules.add_module('a2%i'%l, nn.ReLU())
                modules.add_module('d2%i'%l, nn.Dropout2d(p=.2))
                modules.add_module('m%i'%l, nn.MaxPool2d((2, 1)))
            size[0] = int((size[0]+2*pad-(dil*(kernel-1)+1))/stride+1)
            if (l < n_layers - 1):
                size[0] = size[0] // 2
            size[1] = int((size[1]+2*pad-(dil*(kernel-1)+1))/stride+1)
        modules[-1].unprunable = True
        self.net = modules
        #self.recurrent = nn.GRU(input_size=size[0], hidden_size=size[0] * 4, batch_first=True, bidirectional=False, bias=True)
        #self.recurrent.unprunable = True
        #self.rec_act = nn.ReLU()               
        self.mlp = nn.Sequential()
        """ Then go through MLP """
        for l in range(n_mlp):
            in_s = (l==0) and (size[0] * size[1]) or hidden_size
            out_s = (l == n_mlp - 1) and (hidden_size) or hidden_size
            self.mlp.add_module('h%i'%l, nn.Linear(in_s, out_s))
            if (l < n_mlp - 1):
                self.mlp.add_module('b%i'%l, nn.BatchNorm1d(out_s))
                self.mlp.add_module('a%i'%l, nn.ReLU())
                self.mlp.add_module('d%i'%l, nn.Dropout(p=.25))
        self.mlp[-1].unprunable = True
        """ Set of detectors """
        self.detectors = nn.Sequential()
        for o in range(out_size[0]):
            self.detectors.add_module('d1%i'%o, nn.Linear((hidden_size), (hidden_size)))
            self.detectors.add_module('b1%i'%o, nn.BatchNorm1d((hidden_size)))
            self.detectors.add_module('a1%i'%o, nn.ReLU())
            cur_mod = nn.Linear((hidden_size), out_size[1] * 2)
            cur_mod.unprunable = True
            self.detectors.add_module('d%i'%o, cur_mod)
            self.detectors.add_module('s%i'%o, nn.Sigmoid())
        self.cnn_size = size
    
    def init_parameters(self):
        """ Initialize internal parameters (sub-modules) """
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = x
        for m in range(len(self.net)):
            out = self.net[m](out)
        out = out.reshape(x.shape[0], -1)
        for m in range(len(self.mlp)):
            out = self.mlp[m](out)
        outs = []
        for d in range(len(self.detectors) // 5):
            out_l = self.detectors[5*d+3](self.detectors[5*d+2](self.detectors[5*d+1](self.detectors[5*d](out))))
            outs.append(self.detectors[5*d+4](out_l).reshape(out_l.shape[0], 2, -1).unsqueeze(2))
        out = torch.cat(outs, dim=2)
        return out #torch.cat(outs, dim=2)
    

class SimpleDrummerNet(nn.Module):
    
    def __init__(self, args):
        super(SpectralCNN, self).__init__()
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
class LotteryTranscriptionCNN(LotteryTranscription, SpectralTranscriptionCNN):
    
    def __init__(self, args):
        super(LotteryTranscriptionCNN, self).__init__(args)
        #SpectralTranscriptionCNN.__init__(self, args)
        self.pruning = args.pruning