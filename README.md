## Description
`main.py` is an example script to get predictions from the `StreamingModel` class. Please build your own streaming wrapper around this model.

The function to call for every 3 sec wav audio is `model.predict_3sec(wav)`. Refer to `main.py` for more details.



## Cloning the conda environment

1. `conda env create -f environment.yml`  
2. `conda install pytorch torchvision torchaudio cpuonly -c pytorch`  
3. `conda activate streaming`  

## How to use with your streaming wrapper
`main.py` is an example script for deployment. Note that you have to create your own input streaming wrapper around the model and use its function as you see fit.

The general logic for using the model is this:  

1. Initialize the model globally by `model = StreamingModel(config)`
2. Everytime a new 3 second window comes in, call the function using `model.predict_3sec(wav)`



## Example commands

On a folder 'youtube_test_set' containing wav or mp4 files with top 3 probable classes: `python main.py -d youtube_test_set -k 3 -m A2 -p A2: `
On 1 min wav file with top 5 probable classes: `python main.py --demo test_1min.wav -k 5`     



## Configuration
Most of the values in `config.py` can be kept default, unless you want to tweak the parameters. Note that the `argparser` in `main.py` does not parse for all keywords in `config.py`

```
feature_type: logmelspec
num_bins: 128
gpu: False
sample_rate: 16000
threshold: 0.75
n_fft: 928                                          #formula: (2560*sample_rate)//44100
hop_length: 251                                     #formula: (694*sample_rate)//44100
n_mels: 128
fmin: 20
fmax: 8000
num_frames: 192                                     #formula: int(np.ceil(sample_rate*3/hop_length)) used in resizing spectrogram

# model_loading 
channel_means_path: data/statistics/16.0k/channel_means_logmelspec_012.npy
channel_stds_path: data/statistics/16.0k/channel_stds_logmelspec_012.npy

# prediction config
target_namesA2: ['breaking', 'crowd_scream', 'crying_sobbing', 'explosion', 'gunshot_gunfire', 'motor_vehicle_road','siren', 'speech', 'silence']
k: 3                                               # top-k results to return 
prefix: "A2:"                                          # prefix of each subtitle in the SRT file. eg {M1: chatter, others, breaking}
model: A2
device: cpu
```


### Update 3/3/23 Final deliverable
VAD module is removed and incorporated into the main classifier instead.   
Model `A2` predicts these 9 classes - breaking, crowd_scream, crying_sobbing, explosion, gunshot_gunfire, motor_vehicle_road, siren, speech, silence.  


### Update 23/11/22 hugging face data collection for VAD module
Huggingface has suddenly decided to require users to sign in before being able to use their models. As such, I have included my own READ access token in this repo. For some reason in the future if it expires or fails to work, please regenerate your own access token following the issue [here](https://github.com/pyannote/pyannote-audio/issues/1128). the auth token goes in 'vad.py' under the `PYAN` class

### Update 13/10/22 Alpha release
Includes a new silence detector module. In `api.py` and the docker image, the silence detector is turned on by default.  
In `main.py`, the silence detector is off by default. Use the `-v True` argument to turn it on.  Example command `python main.py --demo test_1min.wav -k 5 -p M3: -m M3 -v True`  
Also includes a new model M3 which is trained on additional speech data.


### Update 07/07/22
Now also creates xml and json captions (same captions as the srt, just different format).   
You can now specify your caption prefix (eg 'M1:') and model to generate the captions in config.   
Example command `python main.py --demo test_1min.wav -k 5 -p M1: -m M1`   

### Update 30/6/22
Now creates srt file for the audio. Load the audio-srt pair using VLC player.   
Now also allows folder input for `-d`. Ignore other files that are not '.mp3', '.mp4' and '.wav'.  

