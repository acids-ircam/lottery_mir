# -*- coding: utf-8 -*-

import librosa
import numpy as np
import pretty_midi as pm
import os
from os import walk
from os.path import join
from natsort import natsorted
import torch
import torch.nn as nn  

# Simplistic GM drum map
drumMap = {
        35:'Kick', 36:'Kick', 37:'Stick', 38:'Snare', 39:'Clap', 40:'Snare',
        41:'Tom', 42:'Hi-Hat', 43:'Tom', 44:'Hi-Hat', 45:'Tom', 46:'Hi-Hat', 
        47:'Tom', 48:'Tom', 49:'Crash', 50:'Tom', 51:'Ride', 52:'Cymbal', 53:'Bell'
        }
# This map simplifies the drum association
simplifyMap = {'Kick':'Kick', 'Stick':'Kick', 'Clap':'Snare', 'Tom':'Snare', 'Crash':'Hi-Hat', 'Ride':'Hi-Hat', 'Bell':'Hi-Hat', 'Cymbal':'Hi-Hat', 'Clap':'Snare'}

# Output dir
output = '/Users/esling/Datasets/mir/drums/'
# Import raw data
drums = ['Kick', 'Snare', 'Hi-Hat']
data_path = '/Users/esling/Datasets/waveform/drums/compiled/small/'
# First load all possible drum sounds
drum_sounds = {}
for d in drums:
    cur_path = data_path + '/' + d
    print('Loading ' + d)
    file_names = []
    for root, dirs, files in walk(cur_path, topdown=False):
        file_names += natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])
    drum_sounds[d] = file_names
# Then load all possible midi files
midi_root = output + 'midi/'
midi_names = []
for root, dirs, files in walk(midi_root, topdown=False):
    midi_names += natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])

#%%
"""

Main function for synthesizing a wave file from MIDI with different paths

"""
def synthesizeMidi(midiFile, outFile, sounds, out_txt):
    annote = {'Snare':'SD', 'Hi-Hat':'HH', 'Kick':'KD', 'Clap':'CL'}
    # Lower sampling rate for size
    sr = 22050
    # loading given MIDI file
    try:
        pmFile = pm.PrettyMIDI(midiFile)
    except:
        return
    # Get the list of drum
    if (pmFile.instruments is None or len(pmFile.instruments) < 1):
        return
    drumPart = pmFile.instruments[0]
    notes = drumPart.notes
    # Find unique pitches
    pitches = []
    for n in notes:
        pitches.append(n.pitch)
        #if (drumMap.get(n.pitch) is None):
        #    print('[!] Unknown drum map - Exiting [!]')
            #return
    # Create an empty out signal
    outSig = np.zeros(int((pmFile.get_end_time() + 6) * sr))
    annotations = []
    n_final = 0
    n_types = []
    last_zero = 0
    for n in notes:
        if (drumMap.get(n.pitch) is None):
            continue
        curInstru = drumMap[n.pitch]
        if (sounds.get(curInstru) is None):
            curInstru = simplifyMap[curInstru]
        n_final += 1
        if (not curInstru in n_types):
            n_types.append(curInstru)
        # Retrieve the sound path
        curSound = sounds[curInstru]
        # Current position in path
        sStart = int(n.start * sr)
        sEnd = sStart + len(curSound)
        # Find the associated sound
        outSig[sStart:sEnd] += curSound
        last_zero = sEnd
        annotations.append('%f\t%s\n'%(n.start, annote[curInstru]))
    if (n_final < 8 or n_final > 20 or len(n_types) < 2):
        return
    outSig = outSig[:last_zero]
    # Create the wav file
    librosa.output.write_wav(outFile, outSig, sr=sr)
    out_file = open(out_txt, 'w')
    for a in annotations:
        out_file.write(a)
    out_file.close()
    print(outFile)
    #for k, v in sepSigs.items():
    #    print(' - Rendering ' + k)
    #    librosa.output.write_wav(outFile + '_' + k + '.wav', v, sr=sr)

# Take a PCA version of the Z_array

# Find all drum loops
for file in midi_names:
    snd_map = {}
    for d in drums:
        cur_drums = drum_sounds[d]
        id_x = np.random.randint(0, len(cur_drums))
        cur_snd, sr = librosa.core.load(cur_drums[id_x])
        snd_map[d] = cur_snd
    base_file = os.path.basename(file).split('.')[0].replace(' ', '_')
    out_file = output + '/audio/' + base_file + '.wav'
    out_txt = output + '/annotations/' + base_file + '.txt'
    synthesizeMidi(file, out_file, snd_map, out_txt)
    
    
#%%
"""
Post-processing the dataset into a trainable format
"""
import matplotlib.pyplot as plt

max_points = 50
n_fft = 2048
n_mels = 64
sr = 22050
# Compute mel filterbank
mel_fb = torch.from_numpy(
    librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
    fmin=20.0, fmax = sr // 4.0))
# Compute mel spectrogram
def compute_melgram(src):
    win = torch.hann_window(n_fft)
    mag_specgrams = torch.stft(src, n_fft, hop_length = n_fft // 2, window=win).pow(2).sum(-1)  # (*, freq, time)
    melgrams = torch.matmul(mel_fb, mag_specgrams)  # (*, mel_freq, time)
    return torch.log(melgrams + 1e-5)

output = '/Users/esling/Datasets/mir/drums/'
annote_root = output + 'annotations/'
audio_root = output + 'audio/'
final_root = output + 'data/'
annotation_names = []
nb_processed = 0
for root, dirs, files in walk(annote_root, topdown=False):
    annotation_names += natsorted([join(root, file_name) for file_name in files if not file_name.startswith('.')])
for a in annotation_names:
    f_id = open(a, 'r')
    audio_file = a.replace('.txt', '.wav').replace('annotations', 'audio')
    final_file = a.replace('.txt', '.npy').replace('annotations', 'data')
    sig, sr = librosa.core.load(audio_file)
    sig = torch.from_numpy(sig)
    onsets = {}
    for line in f_id:
        vals = line.split('\t')
        cur_label = vals[1][:-1]
        if (onsets.get(cur_label) is None):
            onsets[cur_label] = []
        onsets[cur_label].append(float(vals[0]))
    cur_mel = compute_melgram(sig)
    if (cur_mel.shape[1] < max_points):
        continue
    n_steps = cur_mel.shape[1]
    len_sig = len(sig) / sr
    final_onsets = {}
    for k in ['KD', 'SD', 'HH']:
        final_onsets[k] = np.zeros(max_points)
    for k, v in onsets.items():
        cur_vect = np.zeros(n_steps)
        for s in v:
            cur_pos = int((s / len_sig) * n_steps)
            if (cur_pos >= n_steps):
                continue
            cur_vect[cur_pos] = 1
        final_onsets[k] = cur_vect[:max_points]
    np.save(final_file, {'data':cur_mel[:, :max_points], 'labels':final_onsets})
    
    