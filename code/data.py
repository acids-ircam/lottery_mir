# -*- coding: utf-8 -*-

"""
####################

# Data import

# Defines the toy datasets and major data imports

# author    : Philippe Esling
             <esling@ircam.fr>

####################
"""

import torch
import torch.utils.data as data
from  torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import librosa 
import numpy as np
import os
from os import walk
from os.path import join
from natsort import natsorted
import argparse
import copy
import udls
import math
import pretty_midi

from os import path

#%%
"""
###################
Dataloader for waveform classification
###################
"""

class WaveformClassificationLoader(Dataset):
          
    def __init__(self, path, type_d):
        super().__init__()
        self.type = type_d
        self.file_names = [[]] * 3
        types = ['train', 'test', 'valid']
        for t in range(len(types)):
            for root, dirs, files in walk(path + '/' + types[t], topdown=False):
                file_names = natsorted(
                    [join(root, file_name) for file_name in files if not file_name.startswith('.')]
                    )
                if file_names:
                    self.file_names[t] += file_names
        # Join train and valid set (will be re-split by loaders)
        self.file_names[0] += self.file_names[2]
        if (type_d == 'instrument'):
            self.file_names[0] = self.file_names[0][:12000]
            self.file_names[1] = self.file_names[1][:2000]
        
    def set_split(self, name):
        if (name == 'train'):
            self.file_names = self.file_names[0]
        elif (name == 'test'):
            self.file_names = self.file_names[1]
        else:
            print("Unknown split " + name + ".\n")
            exit()   

    def __getitem__(self, index):
        file = os.path.basename(self.file_names[index])
        target = int(file.split('.')[0].split('_')[-1])
        seq = np.load(self.file_names[index], allow_pickle=True)
        audio = torch.from_numpy(seq).float()
        return audio, target

    def __len__(self):
        return len(self.file_names)
    
    
#%%
"""
###################
Dataloader for drums datasets
###################
"""

class WaveformDrumsLoader(Dataset):
          
    def __init__(self, path, type_d='drums', test_percent=.2):
        super().__init__()
        self.type_d = type_d
        self.file_names = []
        self.audio_path = []
        for root, dirs, files in walk(path, topdown=False):
            file_names = natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])
            if file_names:
                self.file_names += file_names
        # Create partition
        indices = list(range(len(self.file_names)))
        split = int(np.floor(test_percent * len(self.file_names)))
        # Shuffle examples
        np.random.shuffle(indices)
        # Split the trainset to obtain a test set
        self.train_idx, self.test_idx = indices[split:], indices[:split]
        
    def set_split(self, name):
        if (name == 'train'):
            cur_idx = self.train_idx
        elif (name == 'test'):
            cur_idx = self.test_idx
        else:
            print("Unknown split " + name + ".\n")
            exit()   
        self.file_names = [self.file_names[i] for i in cur_idx]

    def __getitem__(self, index):
        f_data = np.load(self.file_names[index], allow_pickle=True).item()
        data = f_data['data']
        labels = f_data['labels']
        lab_names = ['KD', 'SD', 'HH']
        target = torch.zeros(3, len(labels[lab_names[0]]))
        for l in range(len(lab_names)):
            target[l] = torch.from_numpy(labels[lab_names[l]])
        if (self.type_d == 'onsets'):
            target = ((target.sum(dim=0) > 0) * 1).unsqueeze(0)
        return data, target.long()

    def __len__(self):
        return len(self.file_names)


"""
###################
Create points for two gaussians
###################
"""
def two_gaussians(args):
    variance = [[.5, 0.1], [0.1, .5]]
    means = [[2, 2], [-2 , -2]]
    # Create two Gaussians
    pts_0 = np.random.multivariate_normal(means[0], variance, int(args.toy_points / 2))
    pts_1 = np.random.multivariate_normal(means[1], variance, int(args.toy_points / 2))
    return pts_0, pts_1

"""
###################
Create points for four planes
###################
"""
def sign_planes(args):
    radius = .5#.1;
    def get_label(point):
        return (np.sign(point[0]) == np.sign(point[1])) * 1
    pts_0, pts_1 = [], []
    for i in range(args.toy_points):
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        noise_x = np.random.uniform(-radius, radius) * args.toy_noise
        noise_y = np.random.uniform(-radius, radius) * args.toy_noise
        if (get_label([x + noise_x, y + noise_y]) == 1):
            pts_1.append([x, y])
        else:
            pts_0.append([x, y])
    return pts_0, pts_1

"""
###################
Create points for XOR problem
###################
"""
def xor_data(args):
    def xor_label(point):
        return point[0] * point[1] >= 0
    pts_0, pts_1 = [], []
    for i in range(args.toy_points):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        x += (((x > 0) * 1) - .5)
        y += (((y > 0) * 1) - .5)
        noise_x = np.random.uniform(-5, 5) * args.toy_noise
        noise_y = np.random.uniform(-5, 5) * args.toy_noise
        if (xor_label([x + noise_x, y + noise_y])):
            pts_1.append([x, y])
        else:
            pts_0.append([x, y])
    return pts_0, pts_1

"""
###################
Create points for two intertwined circles
###################
"""
def circles(args):
    radius = 5
    def get_circle_label(p, center):
        return (np.sqrt(np.sum(p**2 - center**2)) < (radius * 0.5)) * 1
    pts_0, pts_1 = [], []
    # Generate points inside the circle
    for i in range(int(args.toy_points / 2)):
        r = np.random.uniform(0, radius * .5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * args.toy_noise
        noise_y = np.random.uniform(-radius, radius) * args.toy_noise
        #get_circle_label([x + noise_x, y + noise_y], [0, 0])
        pts_0.append([x + noise_x, y + noise_y])
    # Generate points outside the circle
    for i in range(int(args.toy_points / 2)):
        r = np.random.uniform(radius * .7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * args.toy_noise
        noise_y = np.random.uniform(-radius, radius) * args.toy_noise
        #get_circle_label([x + noise_x, y + noise_y], [0, 0])
        pts_1.append([x + noise_x, y + noise_y])
    return pts_0, pts_1

"""
###################
Create points for two intertwined spirals
###################
"""
def two_spirals(args):
    def spiral(delta):
        points = []
        n = args.toy_points
        for i in range(0, args.toy_points):
            r = i / n * 5;
            t = 1.75 * i / n * 2 * np.pi + delta;
            x = r * np.sin(t) + np.random.uniform(-1, 1) * (args.toy_noise * .1)
            y = r * np.cos(t) + np.random.uniform(-1, 1) * (args.toy_noise * .1)
            points.append([x, y])
        return points
    pts_0 = spiral(0)
    pts_1 = spiral(np.pi)
    return pts_0, pts_1

"""
###################
Generate the toy dataset
###################
"""
def generate_toy(args):
    # Generate the rightful set of points
    func_p = {'toy_gauss':two_gaussians, 
              'toy_planes':sign_planes,
              'toy_xor':xor_data,
              'toy_circles':circles,
              'toy_spiral':two_spirals}[args.dataset]
    pts_0, pts_1 = func_p(args)
    # Concatenate all points
    points = np.concatenate((pts_0, pts_1))
    # Generate labels
    labels = np.concatenate((np.zeros(len(pts_0)), np.ones(len(pts_1))), axis=0)
    # Shuffle the points
    idx = np.random.permutation(args.toy_points * 2)
    points = points[idx]
    labels = labels[idx]
    train_set = data.TensorDataset(torch.from_numpy(points[:args.toy_points]).float(), torch.from_numpy(labels[:args.toy_points]).long())
    test_set = data.TensorDataset(torch.from_numpy(points[args.toy_points:]).float(), torch.from_numpy(labels[args.toy_points:]).long())
    return train_set, test_set

"""
###################

Import dataset function

###################
"""
# Main data import
def import_dataset(args):
    # Final dataset directory
    final_dir = args.datadir + args.dataset
    # Main transform
    transform = transforms.Compose([transforms.ToTensor()])
    base_path = args.datadir
    # Retrieve correct data loader
    if args.dataset == "mnist":
        args.input_size = [1, 28, 28]
        args.output_size = 10
        trainset = datasets.MNIST(final_dir, train=True, download=True, transform=transform)
        testset = datasets.MNIST(final_dir, train=False, transform=transform)
    elif args.dataset == "mnist_sub":
        args.input_size = [1, 28, 28]
        args.output_size = 10
        trainset = datasets.MNIST(args.datadir + 'mnist', train=True, download=True, transform=transform)
        testset = datasets.MNIST(args.datadir + 'mnist', train=False, transform=transform)
        trainset.train_data = trainset.train_data[:1000]
        trainset.train_labels = trainset.train_labels[:1000]
        testset.test_data = testset.test_data[:1000]
        testset.test_labels = testset.test_labels[:1000]
    elif args.dataset == "cifar10":
        args.input_size = [3, 32, 32]
        args.output_size = 10
        trainset = datasets.CIFAR10(final_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(final_dir, train=False, transform=transform)      
    elif args.dataset == "fashion_mnist":
        args.input_size = [1, 28, 28]
        args.output_size = 10
        trainset = datasets.FashionMNIST(final_dir, train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(final_dir, train=False, transform=transform)
    elif args.dataset == "cifar100":
        args.input_size = [3, 32, 32]
        args.output_size = 100
        trainset = datasets.CIFAR100(final_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(final_dir, train=False, transform=transform)       
    elif args.dataset[:3] == "toy":
        args.input_size = [2]
        args.output_size = 2
        trainset, testset = generate_toy(args)
    elif args.dataset in ['instrument', 'singing']:
        trainset = WaveformClassificationLoader(final_dir, args.dataset)
        testset = copy.deepcopy(trainset)
        trainset.set_split('train')
        testset.set_split('test')
        if (args.dataset == 'instrument'):
            args.input_size = [1, 32768]
            args.output_size = [13]
        if (args.dataset == 'singing'):
            args.input_size = [1, 132300]
            args.output_size = [10]
    elif args.dataset in ['drums', 'onsets']:
            args.input_size = [64, 50]
            trainset = WaveformDrumsLoader(final_dir + '/data', type_d = args.dataset)
            testset = copy.deepcopy(trainset)
            trainset.set_split('train')
            testset.set_split('test')
            if (args.dataset == 'drums'):
                args.output_size = [3, 50]
            else:
                args.output_size = [1, 50]
    else:
        print("Unknown dataset " + args.dataset + ".\n")
        exit()
    # Compute indices
    indices = list(range(len(trainset)))
    split = int(np.floor(args.valid_size * len(trainset)))
    # Shuffle examples
    np.random.shuffle(indices)
    # Split the trainset to obtain a validation set
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # Create all of the loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.nbworkers, drop_last=True, sampler=train_sampler, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.nbworkers, drop_last=True, sampler=valid_sampler, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.nbworkers, drop_last=True, shuffle=False, pin_memory=False)
    return train_loader, valid_loader, test_loader, args


"""
###################
Test functions
###################
"""
if __name__=="__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    #parser.add_argument("--toy_points",     default=1000,       type=int,   help='')
    #parser.add_argument("--toy_noise",      default=2,         type=int,   help='')
    args = parser.parse_args()
    # List of toy datasets
    args.input_size = [1,64000]
    args.output_size = [1,64000]
    final_dir = '/Users/esling/Datasets/mir/drums_synth/'
    trainset = WaveformDrumsLoader(final_dir, type_d='onset')
    testset = copy.deepcopy(trainset)
    trainset.set_split('train')
    testset.set_split('test')
    # Compute indices
    indices = list(range(len(trainset)))
    split = int(np.floor(.1 * len(trainset)))
    # Shuffle examples
    np.random.shuffle(indices)
    # Split the trainset to obtain a validation set
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # Create all of the loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, drop_last=True, sampler=train_sampler, pin_memory=True)
    #%%
    data = next(iter(train_loader))
    print(data)
    """
    func_p = [two_gaussians, sign_planes, xor_data, circles, two_spirals]
    for f in func_p:
        pts_0, pts_1 = f(args)
        plt.figure()
        pts_plot = np.concatenate(pts_0).reshape(len(pts_0), 2)
        plt.scatter(pts_plot[:, 0], pts_plot[:, 1], color='r')
        pts_plot = np.concatenate(pts_1).reshape(len(pts_1), 2)
        plt.scatter(pts_plot[:, 0], pts_plot[:, 1], color='g')
    """
    