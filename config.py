'''
__author__ = "Andrew Koh Jin Jie, and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Andrew Koh"
__email__ = "andr0081@ntu.edu.sg"
__status__ = "Development"
'''
import numpy as np

feature_type = 'logmelspec'
num_bins = 128
patience = 5

gpu = False
channels = 2

sample_rate = 16000
threshold = 0.9
n_fft=(2560*sample_rate)//44100
hop_length=(694*sample_rate)//44100
n_mels=128
fmin=20
fmax=sample_rate/2
# num_frames = 200
num_frames = int(np.ceil(sample_rate*3/hop_length))

permutation = [0, 1, 2, 3, 4]
target_names = ['breaking', 'chatter', 'crying_sobbing', 'emergency_vehicle', 'explosion', 'gunshot_gunfire', 'motor_vehicle_road', 'screaming', 'siren', 'others']
num_classes = len(target_names)
