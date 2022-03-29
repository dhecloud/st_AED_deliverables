'''
__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"
'''

import torch 
from augmentation.SpecTransforms import ResizeSpectrogram
import numpy as np
from utils import Task5Model, getSampleRateString
import librosa


class StreamingModel():
    '''Master class for generating outputs per 3 sec audio.

    Args:
        p : list of parameters from argparse

    '''
    def __init__(self, p):
        #save parameters
        self.p = p

        #load model
        self.model = Task5Model(p.num_classes).to(p.device)
        model_path = './model/{}k/model_{}_{}'.format(p.sample_rate/1000, p.feature_type, str(p.permutation[0])+str(p.permutation[1])+str(p.permutation[2]))
        self.load_model(model_path)

        #load preprocessing stuff
        self.channel_means = np.load('./data/statistics/{}/channel_means_{}_{}.npy'.format(getSampleRateString(
        p.sample_rate), p.feature_type, str(p.permutation[0])+str(p.permutation[1])+str(p.permutation[2]))).reshape(1, -1, 1)
        self.channel_stds = np.load('./data/statistics/{}/channel_stds_{}_{}.npy'.format(getSampleRateString(
        p.sample_rate), p.feature_type, str(p.permutation[0])+str(p.permutation[1])+str(p.permutation[2]))).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold


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



    def predict_3sec(self, input_wav):
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
            outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
            self.buffer.append(outputs)


        return outputs



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