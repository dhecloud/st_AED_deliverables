'''
__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Andrew Koh"
__email__ = "andr0081@ntu.edu.sg"
'''

import torch
from augmentation.SpecTransforms import ResizeSpectrogram
import numpy as np
from utils import Task5Model, Task5Modelb, getSampleRateString
import librosa


class StreamingM1():
    '''Master class for generating outputs per 3 sec audio.

    Args:
        p : list of parameters from argparse

    '''
    def __init__(self, p):
        #save parameters
        self.p = p

        #load model
        self.model = Task5Model(p.num_classes).to(p.device)
        self.model_path = p.model_path
        self.load_model(self.model_path)

        #load preprocessing stuff
        self.channel_means = np.load(p.channel_means_path).reshape(1, -1, 1)
        self.channel_stds = np.load(p.channel_stds_path).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold
        self.labels = p.target_names

        #buffer init
        self.buffer = []

    def load_model(self, path):
        ''' loads model checkpoint into Task5Model at self.model

        input
            path: path to checkpoint

        output
            None


        '''
        self.model.load_state_dict(torch.load(path, map_location=self.p.device))
        self.model.eval()



    def predict_3sec(self, input_wav, k=1):
        '''takes in wav input and returns the prediction

        args:
            input_wav: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]

        output:
            predictions: tensor containing confidence probability for each class

        Example:
            wav = librosa.load(wav, sr=sample_rate)[0]
            predictions: model.predict_3secs(input_wav)
        '''

        melspec = self.preprocess(input_wav).to(self.p.device)
        with torch.no_grad():
            outputs = self.model(melspec)
            outputs = torch.sigmoid(outputs)[0]
            self.buffer.append(outputs)

        labels = self.return_topk_labels(outputs, k)
        return labels

    def return_topk_labels(self, outputs, k):
        ''' converts predicted values to labels and returns the mots k probable
        '''
        values, indices = torch.topk(outputs, k=k)
        labels = [self.labels[i] for i in indices]
        return labels


    def preprocess(self, input_wav):
        ''' Does preprocessing of the wav input.
            Specifically converts wav to a logmel spectogram and normalizes the spectogram

        Args:
            input: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]

        output:
            logmel: torch tensor of shape (1, 1, mel_bins, seq_len)

        Example:
            wav = librosa.load(wav, sr=sample_rate)[0]
            log_mel model.preprocess(input_wav)

        notes:
            will be used in predict_3sec, but u can use this function on its own for other purposes.

        '''

        melspec = librosa.feature.melspectrogram(
            input_wav,
            sr=self.p.sample_rate,
            n_fft=self.p.n_fft,
            hop_length=self.p.hop_length,
            n_mels=self.p.n_mels,
            fmin=self.p.fmin,
            fmax=self.p.fmax)

        sample = librosa.core.power_to_db(melspec)
        sample = self.resizeSpec(sample)
        sample = torch.Tensor((sample-self.channel_means)/self.channel_stds)
        if len(sample.shape) <= 3:
            sample = torch.unsqueeze(sample, 0)
        return sample


    def clear_buffer(self):
        self.buffer = []

    def reload_params(self,p):
        self.p = p

class StreamingM2():
    '''Master class for generating outputs per 3 sec audio.

    Args:
        p : list of parameters from argparse

    '''
    def __init__(self, p):
        #save parameters
        self.p = p

        #load model
        self.model = Task5Modelb(p.num_classes).to(p.device)
        self.model_path = "model/trying_mv2_w_fdy_head/16.0k/model_logmelspec_012_mobilenetv2_use_cbam_False"
        self.load_model(self.model_path)

        #load preprocessing stuff
        self.channel_means = np.load(p.channel_means_path).reshape(1, -1, 1)
        self.channel_stds = np.load(p.channel_stds_path).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold
        self.labels = p.target_names

        #buffer init
        self.buffer = []

    def load_model(self, path):
        ''' loads model checkpoint into Task5Model at self.model

        input
            path: path to checkpoint

        output
            None


        '''
        self.model.load_state_dict(torch.load(path, map_location=self.p.device))
        self.model.eval()



    def predict_3sec(self, input_wav, k=1):
        '''takes in wav input and returns the prediction

        args:
            input_wav: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]

        output:
            predictions: tensor containing confidence probability for each class

        Example:
            wav = librosa.load(wav, sr=sample_rate)[0]
            predictions: model.predict_3secs(input_wav)
        '''

        melspec = self.preprocess(input_wav).to(self.p.device)
        with torch.no_grad():
            outputs = self.model(melspec)
            outputs = torch.sigmoid(outputs)[0]
            self.buffer.append(outputs)

        labels = self.return_topk_labels(outputs, k)
        return labels

    def return_topk_labels(self, outputs, k):
        ''' converts predicted values to labels and returns the mots k probable
        '''
        values, indices = torch.topk(outputs, k=k)
        labels = [self.labels[i] for i in indices]
        return labels


    def preprocess(self, input_wav):
        ''' Does preprocessing of the wav input.
            Specifically converts wav to a logmel spectogram and normalizes the spectogram

        Args:
            input: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]

        output:
            logmel: torch tensor of shape (1, 1, mel_bins, seq_len)

        Example:
            wav = librosa.load(wav, sr=sample_rate)[0]
            log_mel model.preprocess(input_wav)

        notes:
            will be used in predict_3sec, but u can use this function on its own for other purposes.

        '''

        melspec = librosa.feature.melspectrogram(
            input_wav,
            sr=self.p.sample_rate,
            n_fft=self.p.n_fft,
            hop_length=self.p.hop_length,
            n_mels=self.p.n_mels,
            fmin=self.p.fmin,
            fmax=self.p.fmax)

        sample = librosa.core.power_to_db(melspec)
        sample = self.resizeSpec(sample)
        sample = torch.Tensor((sample-self.channel_means)/self.channel_stds)
        if len(sample.shape) <= 3:
            sample = torch.unsqueeze(sample, 0)
        return sample


    def clear_buffer(self):
        self.buffer = []

    def reload_params(self,p):
        self.p = p
