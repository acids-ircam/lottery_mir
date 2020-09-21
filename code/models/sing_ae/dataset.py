from  torch.utils.data import Dataset
from os import walk
from os.path import join
from natsort import natsorted
import librosa 
import torch
import numpy as np


class Waveform4s_DatasetLoader(Dataset):
          
    def __init__(self, path, sample_rate, test_percent=.2):
        super().__init__()
        self.file_names = []        
        for root, dirs, files in walk(path, topdown=False):
            file_names = natsorted(
                [join(root, file_name) for file_name in files if not file_name.startswith('.')]
            )
            if file_names:
                self.file_names += file_names
        # Create partition
        indices = list(range(len(self.file_names)))
        split = int(np.floor(test_percent * len(self.file_names)))
        # Shuffle examples
        np.random.shuffle(indices)
        # Split the trainset to obtain a test set
        self.train_idx, self.test_idx = indices[split:], indices[:split]
        self.sample_rate = sample_rate
        
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
        (seq, sr) = librosa.load(self.file_names[index], sr=self.sample_rate, mono=True, duration = 4)
        sample_size = 4*self.sample_rate
        if len(seq) < sample_size:
           seq = np.pad(seq,(0,sample_size-len(seq)), 'constant')
        audio = torch.from_numpy(seq).float()
        return audio

    def __len__(self):
        return len(self.file_names)
    
