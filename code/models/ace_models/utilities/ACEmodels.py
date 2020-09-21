#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:16:04 2019

@author: carsault
"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
#from models.ace_models.utilities import ACEdataImport
import pickle
#%%
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, args, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0).to(args.device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

# Convolutional neural network (two convolutional layers)                                                                                                                                                                                                                       
class ConvNet(nn.Module):
    def __init__(self, args, num_classes=25, drop_outRate = 0.6):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            GaussianNoise(args, 0.3),
            nn.Conv2d(1, 16, kernel_size=(25,6), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_outRate),
            nn.MaxPool2d(kernel_size=(3,1), stride=1),
            nn.Conv2d(16, 20, kernel_size=(27,6), stride=1, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop_outRate),
            nn.Conv2d(20, 24, kernel_size=(27,6), stride=1, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop_outRate))
        self.layer5 = nn.Sequential(nn.Linear(11232,200), nn.Linear(200,num_classes))
        
    def forward(self, x):
        out = self.layer1(x)                                                                                                                                                                                                                                               
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        return out
    

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

# Convolutional neural network (two convolutional layers)                                                                                                                                                                                                                       
class LotteryConvNetAce(LotteryModel):
    def __init__(self, args, num_classes=25, drop_outRate = 0.6):
        super(LotteryConvNetAce, self).__init__(args)
        self.pruning = args.pruning
        self.activ = nn.ReLU()
        #layer1
        self.bn0 = nn.BatchNorm2d(1)
        self.gn = GaussianNoise(args, 0.3)
        self.c1 = nn.Conv2d(1, 16, kernel_size=(25,6), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dp1 = nn.Dropout(drop_outRate)
        self.mp1 = nn.MaxPool2d(kernel_size=(3,1), stride=1)
        #layer2
        self.cn2 = nn.Conv2d(16, 20, kernel_size=(27,6), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(20)
        self.dp2 = nn.Dropout(drop_outRate)
        #layer3
        self.cn3 = nn.Conv2d(20, 24, kernel_size=(27,6), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(24)
        self.dp3 = nn.Dropout(drop_outRate)
        self.fl3 = nn.Flatten()
        #layer4
        self.l4 = nn.Linear(1392,200)
        self.bn4 = nn.BatchNorm1d(200)
        self.dp4 = nn.Dropout(drop_outRate)
        #layer5
        self.l5 = nn.Linear(200,num_classes)

        #self.l4.unprunable = True
        self.cn3.unprunable = True
        self.l5.unprunable = True
        
    def forward(self, x):
        out = self.mp1(self.dp1(self.activ(self.bn1(self.c1(self.gn(self.bn0(x)))))))
        out = self.dp2(self.activ(self.bn2(self.cn2(out))))
        out = self.fl3(self.dp3(self.activ(self.bn3(self.cn3(out)))))
        out = self.dp4(self.activ(self.bn4(self.l4(out))))
        out = self.l5(out)                                                                                                                                                                                                                                             
        return out
    
    def train_epoch(self, trainFiles, optimizer, criterion, iteration, args, params):
        self.train()
        train_loss = 0
        for testF in trainFiles:
            #dataset_train = ACEdataImport.createDatasetFull(testF)
            with open(testF, 'rb') as pickle_file:
                dataset_train = pickle.load(pickle_file)
            training_generator = torch.utils.data.DataLoader(dataset_train, pin_memory = True, **params)
            for local_batch, local_labels, local_transp in training_generator:
                # Send the data to device
                local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                # Set gradients to zero
                optimizer.zero_grad()
                # Compute output of the model
                output = self(local_batch)
                # Compute criterion
                loss = criterion(output, local_labels)
                # Run backward
                loss.backward()
                """
                # Call the pruning strategy prior to optimization
                """
                self.pruning.train_callback(self, iteration)
                # Take optimizer step
                optimizer.step()
                train_loss += loss
        return train_loss.item()

    # Function for Testing
    def test_epoch(self, validFiles, criterion, iteration, args, params):
        accuracy = 0
        self.eval()
        with torch.no_grad():
            totSamples = 0
            for validF in validFiles:
                with open(validF, 'rb') as pickle_file:
                    dataset_valid = pickle.load(pickle_file)
                #dataset_valid = None
                #dataset_valid = ACEdataImport.createDatasetFull(testF)
                validating_generator = torch.utils.data.DataLoader(dataset_valid, pin_memory = True, **params)
                for local_batch, local_labels in validating_generator:
                    # Send data to device
                    local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                    # Forward pass the model
                    output = self(local_batch)
                    # Compute max probability index
                    pred = output.data.max(1, keepdim=True)[1]
                    # Compute max accuracy
                    accuracy += pred.eq(local_labels.data.view_as(pred)).sum().item()
                    totSamples += len(local_labels)
            loss = 1. - (accuracy / totSamples)
        return loss
    
    def cumulative_epoch(self, validFiles, optimizer, criterion, limit, args, params):
        accuracy = 0
        self.train()
        # Set gradients to zero
        optimizer.zero_grad()
        totSamples = 0
        full_targets = []
        condition = False
        for validF in validFiles:
            if condition: break
            with open(validF, 'rb') as pickle_file:
                dataset_valid = pickle.load(pickle_file)
            #dataset_valid = None
            #dataset_valid = ACEdataImport.createDatasetFull(testF)
            validating_generator = torch.utils.data.DataLoader(dataset_valid, pin_memory = True, **params)
            for local_batch, local_labels in validating_generator:
                # Send data to device
                local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                # Forward pass the model
                output = self(local_batch)
                loss = criterion(output, local_labels)
                # Compute max probability index
                #pred = output.data.max(1, keepdim=True)[1]
                # Compute max accuracy
                #accuracy += pred.eq(local_labels.data.view_as(pred)).sum().item()
                loss.backward()
                totSamples += len(local_labels)
                if (totSamples > limit):
                    condition = True
                    break
        #loss = 1. - (accuracy / totSamples)
        # Run backward
        
        full_targets.append(local_batch.unsqueeze(1))

        # Add targets
        #return loss
        self.outputs = full_targets
    
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()

# Convolutional neural network (two convolutional layers)                                                                                                                                                                                                                       
class LotteryConvNetAceOLD(LotteryModel):
    def __init__(self, args, num_classes=25, drop_outRate = 0.6):
        super(LotteryConvNetAce, self).__init__(args)
        self.pruning = args.pruning
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            GaussianNoise(args, 0.3),
            nn.Conv2d(1, 16, kernel_size=(25,6), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop_outRate),
            nn.MaxPool2d(kernel_size=(3,1), stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=(27,6), stride=1, padding=0),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop_outRate))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 24, kernel_size=(27,6), stride=1, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop_outRate),
            nn.Flatten())
        self.layer4 = nn.Sequential(
            nn.Linear(1392,200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(drop_outRate))
        self.layer5 = nn.Sequential(
            nn.Linear(200,num_classes)
            )
 
        self.layer3.unprunable = True
        self.layer5.unprunable = True
         
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)                                                                                                                                                                                                                                              
        return out
     
    def train_epoch(self, trainFiles, optimizer, criterion, iteration, args, params):
        self.train()
        train_loss = 0
        for testF in trainFiles:
            #dataset_train = ACEdataImport.createDatasetFull(testF)
            with open(testF, 'rb') as pickle_file:
                dataset_train = pickle.load(pickle_file)
            training_generator = torch.utils.data.DataLoader(dataset_train, pin_memory = True, **params)
            for local_batch, local_labels, local_transp in training_generator:
                # Send the data to device
                local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                # Set gradients to zero
                optimizer.zero_grad()
                # Compute output of the model
                output = self(local_batch)
                # Compute criterion
                loss = criterion(output, local_labels)
                # Run backward
                loss.backward()
                """
                # Call the pruning strategy prior to optimization
                """
                self.pruning.train_callback(self, iteration)
                # Take optimizer step
                optimizer.step()
                train_loss += loss
        return train_loss.item()
 
    # Function for Testing
    def test_epoch(self, validFiles, criterion, iteration, args, params):
        accuracy = 0
        self.eval()
        with torch.no_grad():
            totSamples = 0
            for validF in validFiles:
                with open(validF, 'rb') as pickle_file:
                    dataset_valid = pickle.load(pickle_file)
                #dataset_valid = None
                #dataset_valid = ACEdataImport.createDatasetFull(testF)
                validating_generator = torch.utils.data.DataLoader(dataset_valid, pin_memory = True, **params)
                for local_batch, local_labels in validating_generator:
                    # Send data to device
                    local_batch, local_labels = local_batch.transpose(1,2).view(len(local_batch),1,105,15).to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
                    # Forward pass the model
                    output = self(local_batch)
                    # Compute max probability index
                    pred = output.data.max(1, keepdim=True)[1]
                    # Compute max accuracy
                    accuracy += pred.eq(local_labels.data.view_as(pred)).sum().item()
                    totSamples += len(local_labels)
            loss = 1. - (accuracy / totSamples)
        return loss
     
    def evaluate_epoch(self, test_loader, criterion, iteration, args):
        self.eval()

# Convolutional neural network (two convolutional layers)
class MLP(nn.Module):
    def __init__(self, lenSeq, n_categories, n_hidden, n_latent, decimRatio, n_layer = 1, dropRatio = 0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(int(lenSeq * n_categories / decimRatio), n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(n_layer):
            self.fc2.append(nn.Linear(n_hidden, n_hidden))
            self.bn2.append(nn.BatchNorm1d(n_hidden))
        self.fc3 = nn.Linear(n_hidden, n_latent)
        self.drop_layer = nn.Dropout(p=dropRatio)
        self.n_categories = n_categories
        self.decimRatio = decimRatio
        self.lenSeq = lenSeq
        self.n_layer = n_layer
            
    def forward(self, x):
        x = x.view(-1, int(self.lenSeq * self.n_categories/ self.decimRatio))
        x = F.relu(self.bn1(self.fc1(x)))
        for i in range(self.n_layer):
            x = self.drop_layer(x)
            x = F.relu(self.bn2[i](self.fc2[i](x)))
        x = self.fc3(x)
        return x