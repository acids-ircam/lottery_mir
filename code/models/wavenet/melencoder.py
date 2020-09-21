import torch
import torch.nn as nn
import librosa as li
import numpy as np

module = lambda x: torch.sqrt(x[..., 0]**2 + x[..., 1]**2)


class MelEncoder(nn.Module):
    def __init__(self, sr, hop, n_mel, center=False):
        super().__init__()
        self.hop = hop
        self.nfft = 2048

        mel = li.filters.mel(sr, self.nfft, n_mel, fmin=80)
        mel = torch.from_numpy(mel)

        self.register_buffer("mel", mel)
        self.center = center

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)

        S = torch.stft(x, self.nfft, self.hop, 512, center=self.center)
        S = 2 * module(S) / 512
        S_mel = self.mel.matmul(S)

        if self.training:
            S_mel = S_mel[..., :x.shape[-1] // self.hop]
        return (torch.log10(torch.clamp(S_mel, min=1e-5)) + 5) / 5