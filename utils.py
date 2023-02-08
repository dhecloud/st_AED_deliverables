from typing import Iterator
import pandas as pd
from CBAM import CBAMBlock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from augmentation.SpecTransforms import ResizeSpectrogram
from augmentation.RandomErasing import RandomErasing
from pann_encoder import Cnn10
import os
import random, string

__author__ = "Andrew Koh Jin Jie, Yan Zhen"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

model_archs = ['mobilenetv2', 'pann_cnn10','resnet18','resnet34','resnet50']
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

class ExponentialLayer(nn.Module):
    def __init__(self, bias:bool=True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.randn(1) if bias else torch.zeros(1))
    def forward(self, x):
        return torch.exp(x*(self.weight).to(x.device) + self.bias.to(x.device))


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

class Task5Modelb(nn.Module):

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
        
        self.use_cbam = True


        if self.model_arch == 'mobilenetv2':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
            )
            self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=1280, reduction=16, kernel_size=7)

            self.final = nn.Sequential(
                nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))

        elif self.model_arch == 'mobilenetv3':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
            )
            self.mv3 = torchvision.models.mobilenet_v3_large(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=960, reduction=16, kernel_size=7)

            self.final = nn.Sequential(
                nn.Linear(960, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))


        elif self.model_arch == 'pann_cnn10':
            if len(pann_encoder_ckpt_path) > 0 and os.path.exists(pann_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_encoder_ckpt_path}' does not exist/not found.")
            self.pann_encoder_ckpt_path = pann_encoder_ckpt_path

            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))

            self.encoder = Cnn10()
            if self.pann_encoder_ckpt_path != '':
                self.encoder.load_state_dict(torch.load(
                    self.pann_encoder_ckpt_path)['model'], strict=False)
                print(
                    f'loaded pann_cnn14 pretrained encoder state from {self.pann_encoder_ckpt_path}')

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=512, reduction=16, kernel_size=7)
            
            self.pann_head = nn.Sequential(
                self.cbam if self.use_cbam else nn.Identity(),
                Dynamic_conv2d(512, 256, (1, 1)),
                Dynamic_conv2d(256, 128, (1, 1)),
            )
            # output shape of CNN10 [-1, 512, 39, 4]

            self.final = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, num_classes))

    def forward(self, x):
        if self.model_arch == 'mobilenetv2':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv2.features(x)

        elif self.model_arch == 'mobilenetv3':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv3.features(x)

        elif self.model_arch == 'pann_cnn10':
            x = x  # -> (batch_size, 1, n_mels, num_frames)
            x = x.permute(0, 1, 3, 2)  # -> (batch_size, 1, num_frames, n_mels)
            x = self.AveragePool(x)  # -> (batch_size, 1, num_frames, n_mels/2)
            # try to use a linear layer here.
            x = torch.squeeze(x, 1)  # -> (batch_size, num_frames, 64)
            x = self.encoder(x)
            x = self.pann_head(x)
            
        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)  # -> (batch_size, num_classes)
        return x

class Task5ModelM3(nn.Module):

    def __init__(self, num_classes, model_arch: str = model_archs[0], pann_cnn10_encoder_ckpt_path: str = '', pann_cnn14_encoder_ckpt_path: str = '', use_cbam: bool = False, use_pna: bool = False, use_median_filter: bool = False):
        """Initialising model for Task 5 of DCASE

        Args:
            num_classes (int): Number of classes_
            model_arch (str, optional): Model architecture to be used. One of ['mobilenetv2', 'pann_cnn10', 'pann_cnn14']. Defaults to model_archs[0].
            pann_cnn10_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.
            pann_cnn14_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.

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

        self.use_cbam = use_cbam
        self.use_pna = use_pna
        self.use_median_filter = use_median_filter

        if self.model_arch == 'mobilenetv2':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
                CBAMBlock(
                    channel=3, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size) if self.use_cbam else nn.Identity()
            )
            
            self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=1280, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))

        if self.model_arch == 'mobilenetv3':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
            )
            self.mv3 = torchvision.models.mobilenet_v3_large(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=960, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(960, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))

        elif self.model_arch == 'pann_cnn10':
            if len(pann_cnn10_encoder_ckpt_path) > 0 and os.path.exists(pann_cnn10_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_cnn10_encoder_ckpt_path}' does not exist/not found.")
            self.pann_cnn10_encoder_ckpt_path = pann_cnn10_encoder_ckpt_path

            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))

            self.encoder = Cnn10()
            if self.pann_cnn10_encoder_ckpt_path != '':
                self.encoder.load_state_dict(torch.load(
                    self.pann_cnn10_encoder_ckpt_path)['model'], strict=False)
                print(
                    f'loaded pann_cnn14 pretrained encoder state from {self.pann_cnn10_encoder_ckpt_path}')

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=512, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            if self.use_pna:
                self.pna = ParNetAttention(channel=512)
            
            self.pann_head = nn.Sequential(
                self.cbam if self.use_cbam else nn.Identity(),
                Dynamic_conv2d(512, 256, (1, 1)),
                Dynamic_conv2d(256, 128, (1, 1)),
            )
            # output shape of CNN10 [-1, 512, 39, 4]

            self.final = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, num_classes))

        elif self.model_arch == 'pann_cnn14':
            if len(pann_cnn14_encoder_ckpt_path) > 0 and os.path.exists(pann_cnn14_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_cnn14_encoder_ckpt_path}' does not exist/not found.")
            self.pann_cnn14_encoder_ckpt_path = pann_cnn14_encoder_ckpt_path

            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))

            self.encoder = Cnn14()
            if self.pann_cnn14_encoder_ckpt_path != '':
                self.encoder.load_state_dict(torch.load(
                    self.pann_cnn14_encoder_ckpt_path)['model'], strict=False)
                print(
                    f'loaded pann_cnn10 pretrained encoder state from {self.pann_cnn14_encoder_ckpt_path}')

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=2048, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            if self.use_pna:
                self.pna = ParNetAttention(channel=2048)

            self.final = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, num_classes))

    def forward(self, x):
        if self.model_arch == 'mobilenetv2':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv2.features(x)

        elif self.model_arch == 'mobilenetv3':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv3.features(x)

        elif self.model_arch == 'pann_cnn10' or self.model_arch == 'pann_cnn14':
            x = x  # -> (batch_size, 1, n_mels, num_frames)
            x = x.permute(0, 1, 3, 2)  # -> (batch_size, 1, num_frames, n_mels)
            x = self.AveragePool(x)  # -> (batch_size, 1, num_frames, n_mels/2)
            # try to use a linear layer here.
            x = torch.squeeze(x, 1)  # -> (batch_size, num_frames, 64)
            x = self.encoder(x)
            x = self.pann_head(x)
        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_pna:
            x = self.pna(x)
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)  # -> (batch_size, num_classes)
        return x


class Task5ModelM4(nn.Module):

    def __init__(self, num_classes, model_arch: str = model_archs[0], pann_cnn10_encoder_ckpt_path: str = '', pann_cnn14_encoder_ckpt_path: str = '', use_cbam: bool = False, use_pna: bool = False, use_median_filter: bool = False):
        """Initialising model for Task 5 of DCASE

        Args:
            num_classes (int): Number of classes_
            model_arch (str, optional): Model architecture to be used. One of ['mobilenetv2', 'pann_cnn10', 'pann_cnn14']. Defaults to model_archs[0].
            pann_cnn10_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.
            pann_cnn14_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.

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

        self.use_cbam = use_cbam
        self.use_pna = use_pna
        self.use_median_filter = use_median_filter

        if model_arch.startswith("resnet"):
            self.pools = (
                nn.Identity(),
                nn.AvgPool2d((5,3),stride=(1,1),padding=(2,1)),
                nn.AvgPool2d((7,3),stride=(1,1),padding=(3,1)),
                nn.AvgPool2d((7,5),stride=(1,1),padding=(3,2)),
                nn.MaxPool2d((5,3),stride=(1,1),padding=(2,1)),
                nn.MaxPool2d((7,3),stride=(1,1),padding=(3,1)),
                nn.MaxPool2d((7,5),stride=(1,1),padding=(3,2)),
            )
            input_channels = len(self.pools)
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(input_channels, input_channels, 1, padding=0),
                nn.Dropout(0),
                Dynamic_conv2d(input_channels, 3, 1, padding=0),
                nn.BatchNorm2d(3),
                nn.Identity()
            )
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', model_arch)
            self.mlp = nn.Sequential(
                # nn.Linear(hidden_size*gru_layers, hidden_size*2),
                nn.Linear(1000, 1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )

    def forward(self, x):
        
        if self.model_arch.startswith("resnet"):
            x = torch.cat([pool(x) for pool in self.pools],dim=1)
            x = self.bw2col(x)
            x = self.resnet(x)
            return self.mlp(x)
        
        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_pna:
            x = self.pna(x)
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)  # -> (batch_size, num_classes)
        return x


class Task5ModelM5(nn.Module):

    def __init__(self, num_classes, model_arch: str = model_archs[0], pann_cnn10_encoder_ckpt_path: str = '', pann_cnn14_encoder_ckpt_path: str = '', use_cbam: bool = False, use_pna: bool = False, use_median_filter: bool = False):
        """Initialising model for Task 5 of DCASE

        Args:
            num_classes (int): Number of classes_
            model_arch (str, optional): Model architecture to be used. One of ['mobilenetv2', 'pann_cnn10', 'pann_cnn14']. Defaults to model_archs[0].
            pann_cnn10_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.
            pann_cnn14_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.

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

        self.use_cbam = use_cbam
        self.use_pna = use_pna
        self.use_median_filter = use_median_filter

        if model_arch.startswith("resnet"):
            self.exp_layer = ExponentialLayer(True)
            self.pools = (
                nn.Identity(),
                self.exp_layer,
                nn.AvgPool2d((5,3),stride=(1,1),padding=(2,1)),
                nn.AvgPool2d((7,3),stride=(1,1),padding=(3,1)),
                nn.AvgPool2d((7,5),stride=(1,1),padding=(3,2)),
                nn.MaxPool2d((5,3),stride=(1,1),padding=(2,1)),
                nn.MaxPool2d((7,3),stride=(1,1),padding=(3,1)),
                nn.MaxPool2d((7,5),stride=(1,1),padding=(3,2)),
            )
            input_channels = len(self.pools)
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(input_channels, input_channels, 1, padding=0),
                nn.Dropout(0),
                Dynamic_conv2d(input_channels, 3, 1, padding=0),
                nn.BatchNorm2d(3),
                nn.Identity()
            )
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', model_arch)
            self.mlp = nn.Sequential(
                # nn.Linear(hidden_size*gru_layers, hidden_size*2),
                nn.Linear(1000, 1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 8),
            )

    def forward(self, x):
        
        if self.model_arch.startswith("resnet"):
            x = torch.cat([pool(x) for pool in self.pools],dim=1)
            x = self.bw2col(x)
            x = self.resnet(x)
            return self.mlp(x)
        
        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_pna:
            x = self.pna(x)
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)  # -> (batch_size, num_classes)
        return x

class Dynamic_conv2d(nn.Module):
    """To perform Frequency Dynamic Convolution or Time Dynamic Convolution.
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim="time"):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size if isinstance(
            kernel_size, int) else kernel_size[0]
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes//self.groups, self.kernel_size, self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):  # x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(
                2).unsqueeze(4)    # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(
                2).unsqueeze(3)    # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(
                x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        # size : [n_ker * out_chan, in_chan]
        aggregate_weight = self.weight.view(-1, self.in_planes //
                                            self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                              stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None,
                              stride=self.stride, padding=self.padding, groups=self.groups)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels,
                             self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        # output size : [bs, out_chan, frames, freqs]
        output = torch.sum(output * softmax_attention, dim=1)

        return output

class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(
                in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(
                hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  # x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  # x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  # x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)  # x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)  # x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)  # x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)  # x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)

# from efficientnet_pytorch import EfficientNet
# class EfficientNetClassifier(nn.Module):
#     def __init__(self, backbone=0, n_classes=10):    
#         super().__init__()  
#         self.transform = nn.Linear(1,3)
#         self.mv2 = EfficientNet.from_pretrained('efficientnet-b'+str(backbone), dropout_rate=0.5)
        
#         self.bottleneck = nn.Sequential(
#                 nn.Dropout(0.25),
#                 nn.Linear(1000, 256),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(256),
#                 nn.Dropout(0.25),
#                 nn.Linear(256, 128),
#                 nn.ReLU(),
#                 nn.BatchNorm1d(128))
#         self.classifier = nn.Linear(128, n_classes)
    
#     def forward(self, x):

#         x = x.unsqueeze(-1)
#         x = self.transform(x)
#         x = x.permute(0, 3, 1, 2)
#         x = self.mv2(x)

#         embedding = self.bottleneck(x)
#         x = self.classifier(embedding)
#         x = torch.nn.functional.log_softmax(x, dim=1)
#         return x

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