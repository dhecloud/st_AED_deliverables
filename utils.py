from typing import Iterator
import pandas as pd
from config import feature_type, permutation, sample_rate, num_frames, n_mels
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from augmentation.SpecTransforms import ResizeSpectrogram
from augmentation.RandomErasing import RandomErasing
from pann_encoder import Cnn10
import os
import random, string

__author__ = "Andrew Koh Jin Jie, Yan Zhen"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal", "Anushka Jain"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

model_archs = ['mobilenetv2', 'pann_cnn10']
class_mapping = {}
class_mapping['breaking'] = 0
class_mapping['chatter'] = 1
class_mapping['crying_sobbing'] = 2
class_mapping['emergency_vehicle'] = 3
class_mapping['explosion'] = 4
class_mapping['gunshot_gunfire'] = 5
class_mapping['motor_vehicle_road'] = 6
class_mapping['screaming'] = 7
class_mapping['siren'] = 8
class_mapping['others'] = 9

random_erasing = RandomErasing()

def createRandomFileID():
    result = createStringbyLength(8) + '-' + createStringbyLength(4) + '-' + createStringbyLength(4) + '-' + createStringbyLength(4) + '-' + createStringbyLength(12)
    return result

def createStringbyLength(length):
    choices = string.ascii_lowercase + string.digits
    result = ''.join((random.choice(choices) for x in range(length)))
    return result

def getFileNameFromDf(df: pd.DataFrame, idx: int) -> str:
    """Returns filename for the audio file at index idx of df

    Args:
        df (pd.Dataframe): df of audio files
        idx (int): index of audio file in df

    Returns:
        str: file name of audio file at index 'idx' in df.
    """
    curr = df.iloc[idx, :]
    file_name = curr[0]
    return file_name


def getLabelFromFilename(file_name: str) -> int:
    """Extracts the label from the filename

    Args:
        file_name (str): audio file name

    Returns:
        int: integer label for the audio file name
    """
    label = class_mapping[file_name.split('-')[0]]
    return label


class AudioDataset(Dataset):

    def __init__(self, workspace, df, feature_type=feature_type, perm=permutation, spec_transform=None, image_transform=None, resize=num_frames, sample_rate=sample_rate):

        self.workspace = workspace
        self.df = df
        self.filenames = df[0].unique()
        self.feature_type = feature_type
        self.sample_rate = sample_rate

        self.spec_transform = spec_transform
        self.image_transform = image_transform
        self.resize = ResizeSpectrogram(frames=resize)
        self.pil = transforms.ToPILImage()

        self.channel_means = np.load('{}/data/statistics/{}/channel_means_{}_{}.npy'.format(
            workspace, getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
        self.channel_stds = np.load('{}/data/statistics/{}/channel_stds_{}_{}.npy'.format(
            workspace, getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))

        self.channel_means = self.channel_means.reshape(1, -1, 1)
        self.channel_stds = self.channel_stds.reshape(1, -1, 1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = getFileNameFromDf(self.df, idx)
        labels = getLabelFromFilename(file_name)

        sample = np.load(
            f"{self.workspace}/data/{self.feature_type}/audio_{getSampleRateString(self.sample_rate)}/{file_name}.wav.npy")

        if self.resize:
            sample = self.resize(sample)

        sample = (sample-self.channel_means)/self.channel_stds
        sample = torch.Tensor(sample)

        if self.spec_transform:
            sample = self.spec_transform(sample)

        if self.image_transform:
            # min-max transformation
            this_min = sample.min()
            this_max = sample.max()
            sample = (sample - this_min) / (this_max - this_min)

            # randomly cycle the file
            i = np.random.randint(sample.shape[1])
            sample = torch.cat([
                sample[:, i:, :],
                sample[:, :i, :]],
                dim=1)

            # apply albumentations transforms
            sample = np.array(self.pil(sample))
            sample = self.image_transform(image=sample)
            sample = sample['image']
            sample = sample[None, :, :].permute(0, 2, 1)

            # apply random erasing
            sample = random_erasing(sample.clone().detach())

            # revert min-max transformation
            sample = (sample * (this_max - this_min)) + this_min

        if len(sample.shape) < 3:
            sample = torch.unsqueeze(sample, 0)

        labels = torch.LongTensor([labels]).squeeze()

        data = {}
        data['data'], data['labels'], data['file_name'] = sample, labels, file_name
        return data


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_df):
        self.df = dataset_df
        self.filenames = self.df[0].unique()
        self.length = len(self.filenames)
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, self.length):
            label = self._get_label(idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            if len(self.dataset[label]) > self.balanced_max:
                self.balanced_max = len(self.dataset[label])

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            diff = self.balanced_max - len(self.dataset[label])
            if diff > 0:
                self.dataset[label].extend(
                    np.random.choice(self.dataset[label], size=diff))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self) -> Iterator[int]:
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)

    def _get_label(self, idx):
        file_name = getFileNameFromDf(self.df, idx)
        label = getLabelFromFilename(file_name)
        return label

    def __len__(self):
        return self.balanced_max*len(self.keys)


class Task5Model(nn.Module):

    def __init__(self, num_classes, model_arch: str = model_archs[0], pann_encoder_ckpt_path: str = ''):
        """Initialising model for Task 5 of DCASE

        Args:
            num_classes (int): Number of classes_
            model_arch (str, optional): Model architecture to be used. One of ['mobilenetv2', 'pann_cnn10']. Defaults to model_archs[0].
            pann_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.

        Raises:
            Exception: Invalid model_arch paramater passed.
            Exception: Model checkpoint path does not exist/not found.
        """
        super().__init__()
        self.num_classes = num_classes

        if len(model_arch) > 0:
            if model_arch not in model_archs:
                raise Exception(
                    f'Invalid model_arch={model_arch} paramater. Must be one of {model_archs}')
            self.model_arch = model_arch
            

        if self.model_arch == 'mobilenetv2':
            self.bw2col = nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(), # (128, 656) -> (64, 656) 
                # nn.Conv2d(1, 10, (64, 2), padding=0), nn.ReLU(), # (128, 656) -> (64, 656) 
                nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())
            self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

            self.final = nn.Sequential(
                nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))

        elif self.model_arch == 'pann_cnn10':
            if len(pann_encoder_ckpt_path) > 0 and os.path.exists(pann_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_encoder_ckpt_path}' does not exist/not found.")
            self.pann_encoder_ckpt_path = pann_encoder_ckpt_path
            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))
            self.encoder = Cnn10()
            if self.pann_encoder_ckpt_path!='':
                self.encoder.load_state_dict(torch.load(self.pann_encoder_ckpt_path)['model'], strict = False)
                print(f'loaded pann_cnn10 pretrained encoder state from {self.pann_encoder_ckpt_path}')
            self.final = nn.Sequential(
                nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, num_classes))

    def forward(self, x):
        if self.model_arch == 'mobilenetv2':
            x = self.bw2col(x) # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv2.features(x)
           
        elif self.model_arch == 'pann_cnn10':
            x = x # -> (batch_size, 1, n_mels, num_frames)
            x = x.permute(0, 1, 3, 2) # -> (batch_size, 1, num_frames, n_mels)
            x = self.AveragePool(x) # -> (batch_size, 1, num_frames, n_mels/2) 
            # try to use a linear layer here.
            x = torch.squeeze(x, 1) # -> (batch_size, num_frames, 64)
            x = self.encoder(x)
        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)# -> (batch_size, num_classes)
        return x


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def configureTorchDevice(cuda=torch.cuda.is_available()):
    """Configures PyTorch to use GPU and prints the same.

    Args:
        cuda (bool): To enable GPU, cuda is True, else false. If no value, will then check if GPU exists or not.  

    Returns:
        torch.device: PyTorch device, which can be either cpu or gpu
    """
    device = torch.device('cuda:0' if cuda else 'cpu')
    # print('Device: ', device)
    return device


def getSampleRateString(sample_rate: int):
    """return sample rate in Khz in string form

    Args:
        sample_rate (int): sample rate in Hz

    Returns:
        str: string of sample rate in kHz
    """
    return f"{sample_rate/1000}k"


def dataSampleRateString(type: str, sample_rate: int):
    """Compute string name for the type of data and sample_rate

    Args:
        type (str): type/purpose of data
        sample_rate (int): sample rate of data in Hz

    Returns:
        str: string name for the type of data and sample_rate
    """
    return f"{type}_{getSampleRateString(sample_rate)}"


def set_device(config):
    if config.gpu:
        device = configureTorchDevice()
    else:
        device = configureTorchDevice(False)
    config.device=device
    return config