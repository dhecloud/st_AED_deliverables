'''
__author__ = "Andrew Koh Jin Jie, and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Andrew Koh"
__email__ = "andr0081@ntu.edu.sg"
'''
import librosa
from model import StreamingA2
import numpy as np
import argparse
from utils import set_device
import yaml
from easydict import EasyDict
import srt
from datetime import timedelta
import xml.etree.cElementTree as ET
from xml.dom import minidom
import json
import os
from tqdm import tqdm
import time

with open('config.yaml', "rb") as stream:
    config = EasyDict(yaml.full_load(stream))
parser = argparse.ArgumentParser(description='args for streaming model')
parser.add_argument('-d', '--demo', type=str, default='test_3sec.wav') # path or dir to mp4
parser.add_argument('-n', '--num_frames', type=int, default=config.num_frames)
parser.add_argument('-t', '--threshold', type=float, default=config.threshold)
parser.add_argument('-g', '--gpu', type=bool, default=config.gpu)
parser.add_argument('-k', '--k', type=int, default=config.k)
parser.add_argument('-sr', '--sample_rate', type=int, default=config.sample_rate)
parser.add_argument('-p', '--prefix', type=str, default=config.prefix)
parser.add_argument('-m', '--model', type=str, default=config.model)
args = parser.parse_args()
for k,v in args._get_kwargs():
    config[k] = v

# set device (cpu/gpu) to use
config = set_device(config)
# init model globally
if config.model == 'A2':
    model = StreamingA2(config)
else:
    assert 'model card' == 'not available'

def srts2xml(srts, f_wo_ext):

    #list of subtitles objects. used before the compose function
    split_path = f_wo_ext.split('/') 
    if len(split_path) > 1:
        file_id = split_path[-2]
    else:
        file_id = split_path[-1]
    
    root = ET.Element("AudioDoc", name=file_id)
    doc = ET.SubElement(root, "SoundCaptionList")
    for window_n, sub in enumerate(srts):
        ET.SubElement(doc, "SoundSegment", stime=str(float(window_n)), dur="1.00").text = sub.content

    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(f_wo_ext+'.xml', 'w') as f:
        f.write(xmlstr)

def srts2json(srts, f_wo_ext):

    #list of subtitles objects. used before the compose function
    jsons = []
    for window_n, sub in enumerate(srts):
        jsons.append({"stime": float(window_n), "dur": float(1), "caption": sub.content })
    json_string = json.dumps({"sound":jsons}, indent=4)
    
    with open(f_wo_ext+'.json', 'w') as f:
        f.write(json_string)


def srts2srt(srts, srt_save_path):
    srts = srt.compose(srts, reindex=True)
    with open(srt_save_path,'w') as f:
            f.write(srts) 




def main():
    if os.path.isfile(config.demo):
        paths = [config.demo]
    else:
        print(config.demo)
        assert os.path.isdir(config.demo)
        paths = [os.path.join(config.demo,p) for p in os.listdir(config.demo)]

    for p in tqdm(paths):
        if p[-4:] not in ['.mp3', '.mp4', '.wav']:
            continue
        wav = librosa.load(p , sr=config.sample_rate)[0]
        f_wo_ext = '.'.join(p.split(".")[:-1])
        srt_save_path =  f_wo_ext + '.srt'
        if os.path.exists(srt_save_path):
            with open(srt_save_path,'r') as f:
                lines = f.readlines() 
                lines = '\n'.join(lines)
                # print(lines)
            srts = list(srt.parse(lines))
            append_flag = True
        else:
            srts = []
            append_flag = False

        # for every 3 second wav, pass that segment to predict_3sec
        for curr_window, curr_frame in enumerate(range(0,len(wav), config.sample_rate)):
            
            audio_window = wav[curr_frame:curr_frame+(config.sample_rate*3)]
            cur_window_size = audio_window.shape[0]
            if cur_window_size != config.sample_rate*3:
                pad_len = (config.sample_rate*3) - cur_window_size
                audio_window = np.pad(audio_window, (0, pad_len), mode='constant', constant_values=0 )
            # send 3 sec segment to model for prediction
            predictions, values = model.predict_3sec(audio_window, config.k)
            #prepare formatted srt
            formatted_preds = config.prefix+' '
            for i in range(min(config.k, len(predictions))):
                formatted_preds += f"{i+1}: {predictions[i]} ({values[i]:.2f}) "
            if append_flag == True:
                srts[curr_window].content += '\n'+ formatted_preds
            else:
                srts.append(srt.Subtitle(index=None,start=timedelta(seconds=curr_window+1), end=timedelta(seconds=curr_window+2), content=formatted_preds))
        srts2xml(srts, f_wo_ext)
        srts2json(srts, f_wo_ext)
        srts2srt(srts, srt_save_path)

    # example output predictions: ['chatter', 'others', 'screaming', 'motor_vehicle_road', 'emergency_vehicle']


if __name__ == '__main__':
    main()
