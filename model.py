'''
__author__ = "Andrew Koh Jin Jie, and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Andrew Koh"
__email__ = "andr0081@ntu.edu.sg"
'''

import torch
from augmentation.SpecTransforms import ResizeSpectrogram
import numpy as np
from utils import Task5Model, getSampleRateString, Task5Modelb,Task5ModelM3
 #from utils import EfficientNetClassifier
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
        self.model = Task5Model(len(p.target_namesM1)).to(p.device)
        self.model_path = 'models/M1/16.0k/model_logmelspec_012' 
        self.load_model(self.model_path)

        #load preprocessing stuff
        self.channel_means = np.load(p.channel_means_path).reshape(1, -1, 1)
        self.channel_stds = np.load(p.channel_stds_path).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold
        self.labels = p.target_namesM1

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
            predictions: tensor containing fdence probability for each class

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
    M2 is an improved model that mitigate the dataset class imbalance 

    Args:
        p : list of parameters from argparse

    '''
    def __init__(self, p):
        #save parameters
        self.p = p

        #load model
        self.model = Task5Modelb(len(p.target_namesM2), model_arch="mobilenetv2", pann_encoder_ckpt_path=p.pann_encoder_ckpt_path).to(p.device)
        self.model_path = "models/M2/16.0k/model_logmelspec_012_mobilenetv2_use_cbam_True"
        self.load_model(self.model_path)

        #load preprocessing stuff
        self.channel_means = np.load(p.channel_means_path).reshape(1, -1, 1)
        self.channel_stds = np.load(p.channel_stds_path).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold
        self.labels = p.target_namesM2

        #buffer init
        self.buffer = []
        
    def load_model(self, path):
        ''' loads model checkpoint into Task5Model at self.model

        input
            path: path to checkpoint

        output
            None


        '''
        self.model.load_state_dict(torch.load(path, map_location=self.p.device)['model_state_dict'])
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


class StreamingM3():
    '''Master class for generating outputs per 3 sec audio.

    Args:
        p : list of parameters from argparse

    '''
    def __init__(self, p):
        #save parameters
        self.p = p

        #load model
        self.model = Task5ModelM3(len(p.target_namesM3),model_arch="mobilenetv2").to(p.device)
        self.model_path = 'models/M3/44.1k/model_logmelspec_012_mobilenetv2_use_cbam_False'
        self.load_model(self.model_path)

        #load preprocessing stuff
        self.channel_means = np.load(p.channel_means_path).reshape(1, -1, 1)
        self.channel_stds = np.load(p.channel_stds_path).reshape(1, -1, 1)
        self.resizeSpec = ResizeSpectrogram(frames=p.num_frames)

        #prediction params:
        self.threshold = p.threshold
        self.labels = p.target_namesM3

        #buffer init
        self.buffer = []

    def load_model(self, path):
        ''' loads model checkpoint into Task5Model at self.model

        input
            path: path to checkpoint

        output
            None


        '''
        self.model.load_state_dict(torch.load(path, map_location=self.p.device)['model_state_dict'])
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


# from stft import STFT
# from librosa.util import normalize

# class StreamingM4():
#     '''Master class for generating outputs per 3 sec audio.
#     M4: EfficientNet B0

#     Args:
#         p : list of parameters from argparse

#     '''
#     def __init__(self, p):
        
#         self.MAX_WAV_VALUE = 32768.0
#         self.model = EfficientNetClassifier(0).to(p.device)

#         if p.sample_rate == 16000:
#             self.stft_obj = STFT(filter_length=1024, \
#                 hop_length=512, win_length=1024, \
#                 n_mel_channels=80, \
#                 sampling_rate=16000, \
#                 mel_fmin=0.0, \
#                 mel_fmax=8000, \
#                 window='hann')
            
#             self.model_path = "models/M4/16k/00010000"
#             print("Use 16kHz model")
        
#         elif p.sample_rate == 44100:
#             self.stft_obj = STFT(filter_length=2048, \
#                 hop_length=512, win_length=2048, \
#                 n_mel_channels=128, \
#                 sampling_rate=44100, \
#                 mel_fmin=0.0, \
#                 mel_fmax=12000, \
#                 window='hann')
#             self.model_path = "models/M4/44k/00025000"
#             print("Use 44kHz model")
#         else:
#             assert 'sample_rate' == '44100 or 16000 only'
#         self.p = p
#         self.load_model(self.model_path)

#         #prediction params:
#         self.threshold = p.threshold
#         self.labels = p.target_namesM4
#         #buffer init
#         self.buffer = []
        
#     def load_model(self, path):
#         ''' loads model checkpoint into Task5Model at self.model
#         input
#             path: path to checkpoint
#         output
#             None
#         '''
#         state_dict = torch.load(path, map_location=self.p.device)
#         self.model.load_state_dict(state_dict['classifier'])

#         self.model.eval()

#     def predict_3sec(self, input_wav, k=1):
#         '''takes in wav input and returns the prediction
#         args:
#             input_wav: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]
#         output:
#             predictions: tensor containing confidence probability for each class
#         Example:
#             wav = librosa.load(wav, sr=sample_rate)[0]
#             predictions: model.predict_3secs(input_wav)
#         '''
#         # _, audio= scipy.io.wavfile.read(filename)

#         melspec = self.preprocess(input_wav).to(self.p.device)

#         with torch.no_grad():
#             outputs = self.model(melspec)[0]
#             self.buffer.append(outputs)

#         labels = self.return_topk_labels(outputs, k)
#         return labels

#     def return_topk_labels(self, outputs, k):
#         ''' converts predicted values to labels and returns the mots k probable
#         '''
#         values, indices = torch.topk(outputs, k=k)
#         labels = [self.labels[i] for i in indices]
#         return labels


#     def preprocess(self, input_wav):
#         ''' Does preprocessing of the wav input.
#             Specifically converts wav to a logmel spectogram and normalizes the spectogram
#         Args:
#             input: list containg the values for a 3 sec audio. for eg [0, 0, 0.2, ..., 0.4, 0.4]
#         output:
#             logmel: torch tensor of shape (1, 1, mel_bins, seq_len)
#         Example:
#             wav = librosa.load(wav, sr=sample_rate)[0]
#             log_mel model.preprocess(input_wav)
#         notes:
#             will be used in predict_3sec, but u can use this function on its own for other purposes.
#         '''

#         mel, _ = self.stft_obj.stft(input_wav)
#         if len(mel.size()) == 2:
#             mel = mel.unsqueeze(0)
#         if mel.size(2) == 80 or mel.size(2) == 128:
#             mel = mel.permute(0,2,1)

#         return mel


#     def clear_buffer(self):
#         self.buffer = []

#     def reload_params(self,p):
#         self.p = p