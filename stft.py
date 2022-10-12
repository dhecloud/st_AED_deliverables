import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
import librosa.util as librosa_util
from scipy.io import wavfile as wf

MAX_INT = 32767
MIN_INT = -32768
MAX_WAV_VALUE = 32768.0

mel_basis = None
weight_forward = None
weight_inverse = None
stft_obj = None
h = None
mel_dir = None

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                         n_fft=800, dtype=np.float32, norm=None):
     if win_length is None:
          win_length = n_fft

     n = n_fft + hop_length * (n_frames - 1)
     x = np.zeros(n, dtype=dtype)

     # Compute the squared window at the desired length
     win_sq = get_window(window, win_length, fftbins=True)
     win_sq = librosa_util.normalize(win_sq, norm=norm)**2
     win_sq = librosa_util.pad_center(win_sq, n_fft)

     # Fill the envelope
     for i in range(n_frames):
          sample = i * hop_length
          x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
     return x

def dynamic_range_compression(x, C=1, clip_val=1e-5, log10 = False):
     """
     PARAMS
     ------
     C: compression factor
     """
     if log10:
          return torch.log10(torch.clamp(x, min=clip_val) * C)
     else:
          return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1, log10 = False):
     """
     PARAMS
     ------
     C: compression factor used to compress
     """
     if log10:
          return torch.pow(x, 10) / C
     else:
          return torch.exp(x) / C
          
class STFT(torch.nn.Module):
     """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
     def __init__(self, filter_length=800, hop_length=200, win_length=800, n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                    mel_fmax=8000.0, window='hann'):
          super(STFT, self).__init__()
          self.filter_length = filter_length
          self.hop_length = hop_length
          self.win_length = win_length
          self.window = window
          self.forward_transform = None

          global mel_basis, weight_inverse, weight_forward

          if weight_forward == None or weight_inverse == None:
               scale = self.filter_length / self.hop_length
               fourier_basis = np.fft.fft(np.eye(self.filter_length))

               cutoff = int((self.filter_length / 2 + 1))
               fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                             np.imag(fourier_basis[:cutoff, :])])

               forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
               inverse_basis = torch.FloatTensor(
                    np.linalg.pinv(scale * fourier_basis).T[:, None, :])

               if window is not None:
                    assert(filter_length >= win_length)
                    # get window and zero center pad it to filter_length
                    fft_window = get_window(window, win_length, fftbins=True)
                    fft_window = pad_center(fft_window, filter_length)
                    fft_window = torch.from_numpy(fft_window).float()
                    # window the bases
                    forward_basis *= fft_window
                    inverse_basis *= fft_window

               forward_basis = forward_basis.float()
               inverse_basis = inverse_basis.float()

               weight_forward = Variable(forward_basis, requires_grad=False)
               weight_inverse = Variable(inverse_basis, requires_grad=False)

          if mel_basis == None:
               mel_basis = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
               mel_basis = torch.from_numpy(mel_basis).float()

     def spectral_normalize(self, magnitudes):
          output = dynamic_range_compression(magnitudes)
          return output
     def spectral_de_normalize(self, magnitudes):
          output = dynamic_range_decompression(magnitudes)
          return output
     def transform(self, input_data, train_mode=False):
          num_batches = input_data.size(0)
          num_samples = input_data.size(1)

          self.num_samples = num_samples

          # similar to librosa, reflect-pad the input
          input_data = input_data.view(num_batches, 1, num_samples)

          if train_mode:
               input_data = F.pad(input_data.unsqueeze(1), 
                                   (int((self.filter_length-self.hop_length)/2), 
                                   int((self.filter_length-self.hop_length)/2), 0, 0),
                                   mode='reflect')
          else:
               input_data = F.pad(input_data.unsqueeze(1), 
                                   (int(self.filter_length / 2), 
                                   int(self.filter_length / 2), 0, 0), 
                                   mode='reflect')
          
          input_data = input_data.squeeze(1)

          global weight_forward

          if train_mode:
               
               forward_transform = F.conv1d(
                    input_data,
                    weight_forward.to("cuda"),
                    stride=self.hop_length,
                    padding=0)
          else:
               forward_transform = F.conv1d(
                    input_data,
                    weight_forward,
                    stride=self.hop_length,
                    padding=0)

          cutoff = int((self.filter_length / 2) + 1)
          real_part = forward_transform[:, :cutoff, :]
          imag_part = forward_transform[:, cutoff:, :]
          #print('cutoff: {} real_part: {} filter_length: {}'.format(cutoff, real_part.size(), self.filter_length))

          magnitude = torch.sqrt(real_part**2 + imag_part**2)
          phase = torch.autograd.Variable(
               torch.atan2(imag_part.data, real_part.data))

          return magnitude, phase

     def inverse(self, magnitude, phase):
          recombine_magnitude_phase = torch.cat(
               [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

          global weight_inverse
          
          inverse_transform = F.conv_transpose1d(
               recombine_magnitude_phase,
               weight_inverse,
               stride=self.hop_length,
               padding=0)

          if self.window is not None:
               window_sum = window_sumsquare(
                    self.window, magnitude.size(-1), hop_length=self.hop_length,
                    win_length=self.win_length, n_fft=self.filter_length,
                    dtype=np.float32)
               # remove modulation effects
               approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0])
               window_sum = torch.autograd.Variable(
                    torch.from_numpy(window_sum), requires_grad=False)
               window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
               inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

               # scale by hop ratio
               inverse_transform *= float(self.filter_length) / self.hop_length

          inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
          inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

          return inverse_transform

     def forward(self, input_data):
          self.magnitude, self.phase = self.transform(input_data)
          reconstruction = self.inverse(self.magnitude, self.phase)
          return reconstruction

     def mel_spectrogram_torchstft(self, y, train_mode=False):
          global mel_basis
          
          hann_window = torch.hann_window(self.win_length).to("cuda")
          y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.filter_length-self.hop_length)/2), int((self.filter_length-self.hop_length)/2)), mode='reflect')
          y = y.squeeze(1)
          spec = torch.stft(y, self.filter_length, hop_length=self.hop_length, win_length=self.win_length, window=hann_window,
                              center=False, pad_mode='reflect', normalized=False, onesided=True)
          spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
          spec = torch.matmul(mel_basis.to("cuda"), spec)
          spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
          return spec     

     def mel_spectrogram(self, y, train_mode=False):

          global mel_basis

          magnitudes, phrases = self.transform(y, train_mode)
          magnitudes = magnitudes.data

          if train_mode:
               mel_output = torch.matmul(mel_basis.to("cuda"), magnitudes)
          else:
               mel_output = torch.matmul(mel_basis, magnitudes)
          mel_output = self.spectral_normalize(mel_output)
          
          return mel_output
     def stft(self, audio):
          if isinstance(audio, np.ndarray):
               audio = torch.from_numpy(audio).float()
          else:
               audio = audio.float()
          audio_norm = audio / MAX_WAV_VALUE
          audio_norm = audio_norm.unsqueeze(0)
          audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
          melspec = self.mel_spectrogram(audio_norm)
          melspec = torch.squeeze(melspec, 0)
          return melspec, audio_norm.squeeze(0)
     def convert_spectrogram_format(self, mel):
          mel = self.spectral_de_normalize(mel)
          return dynamic_range_compression(mel, log10 = True)


def get_mel(fi, mel_dir, h, stft_obj, preprocess=False):

     _, audio= wf.read(fi)
     file_save = mel_dir + "/" + fi.split("/")[-1].replace(".wav",".npy")
     if preprocess:
          if len(audio) < 22050:
               print(len(audio))
               return 0

     audio = audio[:int(len(audio)/h.hop_size)*h.hop_size]
     mel, audio_norm = stft_obj.stft(audio)
     
     np.save(file_save, mel)
     return file_save

def get_mel_from_wav(audio,fi, mel_dir, h, stft_obj, preprocess=False):

     file_save = mel_dir + "/" + fi.split("/")[-1].replace(".wav",".npy")
     if preprocess:
          if len(audio) < 22050:
               print(len(audio))
               return 0

     mel, audio_norm = stft_obj.stft(audio)
     
     np.save(file_save, mel)
     return file_save