'''
__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"
'''

from model import StreamingModel
import librosa
import config as config
import argparse
from utils import set_device




config = set_device(config)

def main(args):



    wav = librosa.load('test.wav', sr=config.sample_rate)[0]
    model = StreamingModel(config)
    predictions = model.predict_3sec(wav)
    print(predictions)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='args for streaming model')

    parser.add_argument(
        '-f', '--feature_type', type=str, default=config.feature_type)
    parser.add_argument(
        '-n', '--num_frames', type=int, default=config.num_frames)
    parser.add_argument(
        '-t', '--threshold', type=float, default=config.threshold)
    parser.add_argument('-g', '--gpu', type=bool, default=config.gpu)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=config.permutation)

    args = parser.parse_args()

    main(args)


