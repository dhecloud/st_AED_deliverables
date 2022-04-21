
## Description
`main.py` shows an example script to get predictions from the `StreamingModel` class.  

The main function to call for every 3 sec wav audio is `model.predict_3sec(wav)`. Refer to `main.py` for more details.


## Cloning the conda environment

1. `conda env create -f environment.yml`
2. `conda activate streaming`

## How to use with your streaming wrapper
`main.py` is an example script for deployment. Note that you have to create your own input streaming wrapper around the model.

1. Initialize the model globally by `model = StreamingModel(config)`
2. Everytime a new 3 second window comes in, call the function using `model.predict_3sec(wav)`

## Configuration
Most of the values in `config.py` can be kept default, unless you want to tweak the parameters. Note that the `argparser` in `main.py` does not parse for all keywords in `config.py`

```
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
```

## Example commands

Default: `python main.py`  
changing threshold: `python main.py --threshold 0.09`
