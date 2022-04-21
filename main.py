'''
__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Andrew Koh"
__email__ = "andr0081@ntu.edu.sg"
'''

from model import StreamingModel
import librosa
import argparse
from utils import set_device
import yaml
from easydict import EasyDict

with open('config.yaml', "rb") as stream:
    config = EasyDict(yaml.full_load(stream))
parser = argparse.ArgumentParser(description='args for streaming model')
parser.add_argument('-d', '--demo', type=str, default='test_3sec.wav')
parser.add_argument('-n', '--num_frames', type=int, default=config.num_frames)
parser.add_argument('-t', '--threshold', type=float, default=config.threshold)
parser.add_argument('-g', '--gpu', type=bool, default=config.gpu)
parser.add_argument('-k', '--k', type=int, default=config.k)
args = parser.parse_args()
for k,v in args._get_kwargs():
    config[k] = v
config = set_device(config)
model = StreamingModel(config)


def main():

    # load example wav file
    wav = librosa.load(config.demo , sr=config.sample_rate)[0]

    # for every 3 second wav, pass that segment to predict_3sec
    for curr_window, curr_frame in enumerate(range(0,len(wav), config.sample_rate*3)):
        predictions = model.predict_3sec(wav[curr_frame:curr_frame+(config.sample_rate*3)], config.k)
        print(f"{curr_window}:, {predictions}")
        
    # example output predictions: ['chatter', 'others', 'screaming', 'motor_vehicle_road', 'emergency_vehicle']


if __name__ == '__main__':
    main()
