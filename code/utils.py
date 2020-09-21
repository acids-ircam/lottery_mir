# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def plot_batch_images(batch, name=None):
    # Create one big image for plot
    img = np.zeros((batch.shape[2] * 4 + 3, batch.shape[3] * 4 + 3))
    for b in range(min(batch.shape[0], 16)):
        row = int(b / 4); col = int(b % 4)
        r_p = row * batch.shape[2] + row; c_p = col * batch.shape[3] + col
        img[r_p:(r_p+batch.shape[2]),c_p:(c_p+batch.shape[3])] = batch[b].squeeze()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if (name is not None):
        plt.savefig(name + '.png')
        plt.close()    

def plot_batch_compare(batch, reconstruct, name=None):
    # Create one big image for plot
    img = np.zeros((batch.shape[2] * 4 + 3, batch.shape[3] * 4 + 3))
    for b in range(min(batch.shape[0], 8)):
        row = int(b / 4); col = int(b % 4)
        r_p = row * batch.shape[2] + row; c_p = col * batch.shape[3] + col
        img[r_p:(r_p+batch.shape[2]),c_p:(c_p+batch.shape[3])] = batch[b].squeeze()
    for b2 in range(min(reconstruct.shape[0], 8)):
        b = b2 + 8
        row = int(b / 4); col = int(b % 4)
        r_p = row * batch.shape[2] + row; c_p = col * batch.shape[3] + col
        img[r_p:(r_p+batch.shape[2]),c_p:(c_p+batch.shape[3])] = reconstruct[b2].squeeze()
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    if (name is not None):
        plt.savefig(name + '.png')
        plt.close()
        
def plot_batch_wav(batch, name=None):
    nb_plots = min(batch.shape[0], 16)
    nb_axs = int(np.sqrt(nb_plots))
    fig, axs = plt.subplots(nb_axs,nb_axs,sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0},figsize=(10, 10))
    for i in range(nb_axs):
        for j in range(nb_axs):
            axs[i, j].plot(batch[i+j,:])
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    if (name is not None):
        plt.savefig(name + '.png')
        plt.close()
        
def plot_batch_compare_wav(batch, reconstruct, name=None):
    nb_plots = min(batch.shape[0], 16)
    nb_axs = int(np.sqrt(nb_plots))
    fig, axs = plt.subplots(nb_axs,nb_axs,sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(10, 10))
    
    for i in range(nb_axs):
        for j in range(nb_axs):
            if j%2:
                axs[i, j].plot(batch[i+(j//2),:])
            else:
                axs[i, j].plot(reconstruct[i+(j//2),:])
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    if (name is not None):
        plt.savefig(name + '.png')
        plt.close()

def write_batch_wav(batch, sample_rate, name = None):
    nb_saves = min(batch.shape[0], 16)
    duration = batch.shape[1]
    sounds = np.zeros(nb_saves * duration)
    for i in range(nb_saves):
        sounds[i*duration:(i+1)*duration] = batch[i,:]
    if (name is not None):
        wavfile.write(name + ".wav", sample_rate, sounds.T)
        
def write_batch_compare_wav(batch, reconstruct, sample_rate, name = None):
    nb_saves = min(batch.shape[0], 16)
    duration = batch.shape[1]
    sounds = np.zeros(nb_saves * duration)
    for i in range(nb_saves):
        if i%2:
            sounds[i*duration:(i+1)*duration] = batch[(i//2),:]
        else:
            sounds[i*duration:(i+1)*duration] = reconstruct[(i//2),:]
    if (name is not None):
        wavfile.write(name + ".wav", sample_rate, sounds.T)
        
from  torch.utils.data import Dataset
from os import walk
from os.path import join
from natsort import natsorted
import librosa 
import torch

path = '/fast-1/datasets/waveform/sol-ordinario/audio/'
final_names = []        
for root, dirs, files in walk(path, topdown=False):
    file_names = natsorted(
        [join(root, file_name) for file_name in files if not file_name.startswith('.')]
        )
    if file_names:
        final_names += file_names

for index in range(len(final_names)):
    (seq, sr) = librosa.load(final_names[index], sr=16000, mono=True, duration = 4)
    wavfile.write(final_names[index], sr, seq)
    
