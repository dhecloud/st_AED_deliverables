import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
import random
import torchvision
import librosa

class TimeMask(object):
    """Apply time masking to a given spectrogram

    Args:
        T: Max length of time block to be masked
        num_masks: Number of masks to be applied
        replace_with_zero: If the masked area should be replaced with zeros or with the mean
    """

    def __init__(self, T=40, num_masks=1, replace_with_zero=False):
        # assert isinstance(output_size, (int, tuple))
        self.T = T
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def __call__(self, sample):
        cloned = sample.clone()
        len_spectro = cloned.shape[2]
    
        for i in range(0, self.num_masks):
            t = random.randrange(0, self.T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (self.replace_with_zero): cloned[0][:,t_zero:mask_end] = 0
            else: cloned[0][:,t_zero:mask_end] = cloned.mean()

        return cloned 

class FrequencyMask(object):
    """Apply time masking to a given spectrogram

    Args:
        T: Max length of time block to be masked
        num_masks: Number of masks to be applied
        replace_with_zero: If the masked area should be replaced with zeros or with the mean
    """

    def __init__(self, F=30, num_masks=1, replace_with_zero=False):
        # assert isinstance(output_size, (int, tuple))
        self.F = F
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def __call__(self, sample):
        
        cloned = sample.clone()
        num_mel_channels = cloned.shape[1]
    
        for i in range(0, self.num_masks):        
            f = random.randrange(0, self.F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f) 
            if (self.replace_with_zero): cloned[0][f_zero:mask_end] = 0
            else: cloned[0][f_zero:mask_end] = cloned.mean()

        return cloned

class MelSpectrogram(object):
    """Generate log scaled mel-spectrogram for a given audio waveform

    Args:
        To do
    """

    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax):
        # assert isinstance(output_size, (int, tuple))
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, sample):
        
        melspec = librosa.feature.melspectrogram(sample,
        sr = self.sr,
        n_fft = self.n_fft,
        hop_length = self.hop_length,
        n_mels = self.n_mels,
        fmin = self.fmin,
        fmax = self.fmax)
        logmel = librosa.core.power_to_db(melspec)
        return logmel

class RandomCycle(object):
    """Randomly cycle a given spectrogram"""

    def __init__(self, F=30, num_masks=1, replace_with_zero=False):
        # assert isinstance(output_size, (int, tuple))
        self.F = F
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def __call__(self, sample):

        # randomly cycle the file
        i = np.random.randint(sample.shape[2])
        sample = torch.cat([
            sample[:, :, i:],
            sample[:, :, :i]],
            dim=2)

        return sample

class ResizeSpectrogram(object):
    """Resize spectrogram to a given size"""

    def __init__(self, frames):
        # assert isinstance(output_size, (int, tuple))
        self.frames = frames

    def __call__(self, sample):

        if sample.shape[1]<self.frames:
            padding_len = self.frames-sample.shape[1]
            zero_pad = np.zeros((sample.shape[0], padding_len))
            sample = np.hstack((sample, zero_pad))
        elif sample.shape[1]>self.frames:
            sample = sample[:, :self.frames]
    
        return sample
    
        
 


        