# import collections
# import contextlib
# import sys
# import os, random

# import wave
from random import sample
from pyannote.audio import Pipeline

import torch
import numpy as np
import scipy
import scipy.signal
# from  scipy.io import wavfile
import matplotlib.pyplot as plt
# import pandas as pd

# from pydub import AudioSegment
from tqdm import tqdm


class STE():
    def stride_trick(self, a, stride_length, stride_step):
        """
        apply framing using the stride trick from numpy.

        Args:
            a (array) : signal array.
            stride_length (int) : length of the stride.
            stride_step (int) : stride step.

        Returns:
            blocked/framed array.
        """
        nrows = ((a.size - stride_length) // stride_step) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a,
                                            shape=(nrows, stride_length),
                                            strides=(stride_step*n, n))

    def framing(self, sig, fs=16000, win_len=0.025, win_hop=0.01):
        """
        transform a signal into a series of overlapping frames (=Frame blocking).

        Args:
            sig     (array) : a mono audio signal (Nx1) from which to compute features.
            fs        (int) : the sampling frequency of the signal we are working with.
                            Default is 16000.
            win_len (float) : window length in sec.
                            Default is 0.025.
            win_hop (float) : step between successive windows in sec.
                            Default is 0.01.

        Returns:
            array of frames.
            frame length.

        Notes:
        ------
            Uses the stride trick to accelerate the processing.
        """
        # run checks and assertions
        if win_len < win_hop: print("ParameterError: win_len must be larger than win_hop.")

        # compute frame length and frame step (convert from seconds to samples)
        frame_length = win_len * fs
        frame_step = win_hop * fs
        signal_length = len(sig)
        frames_overlap = frame_length - frame_step

        # compute number of frames and left sample in order to pad if needed to make
        # sure all frames have equal number of samples  without truncating any samples
        # from the original signal
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

        # apply stride trick
        frames = self.stride_trick(pad_signal, int(frame_length), int(frame_step))
        return frames, frame_length

    def _calculate_normalized_short_time_energy(self, frames):
        return np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames)))**2, axis=-1) / len(frames)**2

    def naive_frame_energy_vad(self, sig, fs, threshold=-20, win_len=0.25, win_hop=0.25, E0=1e7):
        # framing
        frames, frames_len = self.framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

        # compute short time energies to get voiced frames
        energy = self._calculate_normalized_short_time_energy(frames)
        log_energy = 10 * np.log10(energy / E0)

        # normalize energy to 0 dB then filter and format
        energy = scipy.signal.medfilt(log_energy, 5)
        energy = np.repeat(energy, frames_len)

        # compute vad and get speech frames
        vad     = np.array(energy > threshold, dtype=sig.dtype)
        vframes = np.array(frames.flatten()[np.where(vad==1)], dtype=sig.dtype)
        return energy, vad, np.array(vframes, dtype=np.float64)

    def multi_plots(self, data, titles, fs, plot_rows, step=1, colors=["b", "r", "m", "g", "b", "y"]):
        # first fig
        # plt.subplots(plot_rows, 1, figsize=(20, 10))
        # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.99, wspace=0.4, hspace=0.99)

        # for i in range(plot_rows):
        #     plt.subplot(plot_rows, 1, i+1)
        #     y = data[i]
        #     plt.plot([i/fs for i in range(0, len(y), step)], y, colors[i])
        #     plt.gca().set_title(titles[i])
        # plt.show()

        # second fig
        sig, vad = data[0], data[-2]
        # plot VAD and orginal signal
        plt.subplots(1, 1, figsize=(20, 10))
        plt.plot([i/fs for i in range(len(sig))], sig, label="Signal")
        plt.plot([i/fs for i in range(len(vad))], max(sig)*vad, label="VAD")
        plt.legend(loc='best')
        plt.show()

    # if __name__ == "__main__":
    def filter(self, audio, sample_rate, threshold=-20):
        energy, vad, voiced = self.naive_frame_energy_vad(audio, sample_rate, threshold=threshold, win_len=0.025, win_hop=0.025)
        return np.array(vad)

# class WEB():
#     def __init__(self):
#         self.bool_list = []
#     # def read_wave(path):
#     #     """Reads a .wav file.
#     #     Takes the path, and returns (PCM audio data, sample rate).
#     #     """
#     #     with contextlib.closing(wave.open(path, 'rb')) as wf:
#     #         num_channels = wf.getnchannels()
#     #         assert num_channels == 1
#     #         sample_width = wf.getsampwidth()
#     #         assert sample_width == 2
#     #         sample_rate = wf.getframerate()
#     #         assert sample_rate in (8000, 16000, 32000, 48000)
#     #         pcm_data = wf.readframes(wf.getnframes())
#     #         return pcm_data, sample_rate

#     def write_wave(self, path, audio, sample_rate):
#         """Writes a .wav file.

#         Takes path, PCM audio data, and sample rate.
#         """
#         with contextlib.closing(wave.open(path, 'wb')) as wf:
#             wf.setnchannels(1)
#             wf.setsampwidth(2)
#             wf.setframerate(sample_rate)
#             wf.writeframes(audio)

#     class Frame(object):
#         """Represents a "frame" of audio data."""
#         def __init__(self, bytes, timestamp, duration):
#             self.bytes = bytes
#             self.timestamp = timestamp
#             self.duration = duration

#     def frame_generator(self, frame_duration_ms, audio, sample_rate):
#         """Generates audio frames from PCM audio data.

#         Takes the desired frame duration in milliseconds, the PCM data, and
#         the sample rate.

#         Yields Frames of the requested duration.
#         """
#         n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
#         offset = 0
#         timestamp = 0.0
#         duration = (float(n) / sample_rate) / 2.0
#         while offset + n < len(audio):
#             yield self.Frame(audio[offset:offset + n], timestamp, duration)
#             timestamp += duration
#             offset += n

#     def vad_collector(self, sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
#         """Filters out non-voiced audio frames.

#         Given a webrtcvad.Vad and a source of audio frames, yields only
#         the voiced audio.

#         Uses a padded, sliding window algorithm over the audio frames.
#         When more than 90% of the frames in the window are voiced (as
#         reported by the VAD), the collector triggers and begins yielding
#         audio frames. Then the collector waits until 90% of the frames in
#         the window are unvoiced to detrigger.

#         The window is padded at the front and back to provide a small
#         amount of silence or the beginnings/endings of speech around the
#         voiced frames.

#         Arguments:

#         sample_rate - The audio sample rate, in Hz.
#         frame_duration_ms - The frame duration in milliseconds.
#         padding_duration_ms - The amount to pad the window, in milliseconds.
#         vad - An instance of webrtcvad.Vad.
#         frames - a source of audio frames (sequence or generator).

#         Returns: A generator that yields PCM audio data.
#         """
#         num_padding_frames = int(padding_duration_ms / frame_duration_ms)
#         # We use a deque for our sliding window/ring buffer.
#         ring_buffer = collections.deque(maxlen=num_padding_frames)
#         # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
#         # NOTTRIGGERED state.
#         triggered = False

#         voiced_frames = []
#         # bool_list = []

#         for frame in frames:
#             is_speech = vad.is_speech(frame.bytes, sample_rate)

#             # sys.stdout.write('1' if is_speech else '0')
#             self.bool_list.append(is_speech)

#             if not triggered:
#                 ring_buffer.append((frame, is_speech))
#                 num_voiced = len([f for f, speech in ring_buffer if speech])
#                 # If we're NOTTRIGGERED and more than 90% of the frames in
#                 # the ring buffer are voiced frames, then enter the
#                 # TRIGGERED state.
#                 if num_voiced > 0.9 * ring_buffer.maxlen:
#                     triggered = True
#                     # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))

#                     # We want to yield all the audio we see from now until
#                     # we are NOTTRIGGERED, but we have to start with the
#                     # audio that's already in the ring buffer.
#                     for f, s in ring_buffer:
#                         voiced_frames.append(f)
#                     ring_buffer.clear()
#             else:
#                 # We're in the TRIGGERED state, so collect the audio data
#                 # and add it to the ring buffer.
#                 voiced_frames.append(frame)
#                 ring_buffer.append((frame, is_speech))
#                 num_unvoiced = len([f for f, speech in ring_buffer if not speech])
#                 # If more than 90% of the frames in the ring buffer are
#                 # unvoiced, then enter NOTTRIGGERED and yield whatever
#                 # audio we've collected.
#                 if num_unvoiced > 0.9 * ring_buffer.maxlen:
#                     # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
#                     triggered = False
#                     yield b''.join([f.bytes for f in voiced_frames])
#                     ring_buffer.clear()
#                     voiced_frames = []
#         # if triggered:
#         #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
#         # sys.stdout.write('\n')

#         # If we have any leftover voiced audio when we run out of input,
#         # yield it.
#         if voiced_frames:
#             yield b''.join([f.bytes for f in voiced_frames])

#     # def smoothen(resolution = 50):
#     #     for 
        
#     def main(self, difficulty, audio):
#         # if len(args) != 2:
#         #     print('argument error')
#         audioname = audio.split('/')[-1]
#         sample_rate, signal = scipy.io.wavfile.read(audio)
#         # signal, sample_rate = read_wave(args[1])
#         vad = webrtcvad.Vad(difficulty)
#         frames = self.frame_generator(30, signal, sample_rate)
#         frames = list(frames)
#         segments = self.vad_collector(sample_rate, 30, 300, vad, frames)
#         complete_sig = bytes()
#         output_file = ''
#         for i, segment in enumerate(segments):
#             complete_sig += segment
#             output_file = 'WEB ' + audioname[:-4] + '-%002d.wav' % (i,)

#         # print(' Writing %s' % (output_file,))
#         # write_wave(output_file, complete_sig, sample_rate)
#         # scipy.io.wavfile.write('STE ' + audio.split('/')[-1],
#         #                        fs,  np.array(complete_sig, dtype=sig.dtype))
#         # display(Audio(output_file))

#         # vad = bool_list
#         # plot VAD and orginal signal
#         # plt.subplots(1, 1, figsize=(20, 10))
#         # plt.plot([i/sample_rate for i in range(len(signal))], signal, label="Signal")
#         # amplitude = max(signal)
#         vad = np.repeat([1 if i else 0 for i in self.bool_list], 100)
#         # plt.step([i*10 for i in vad], [i/33.3 for i in range(3330)])
#         # plt.plot([i*10/len(vad) for i in range(len(vad))], vad, label="VAD")
#         # plt.legend(loc='best')
#         # plt.show()
        
#         # vad = smoothen(vad)
        
#         return vad

#     # if __name__ == '__main__':
#     def filter(self, audio):
#         vad = self.main(3, audio)
#         return np.array(vad)

class PYAN():
    def __init__(self):
        # self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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")
        self.pipeline = Pipeline.from_pretrained("cache/pyannote/models--pyannote--voice-activity-detection/snapshots/f3b1193d5288ed24de9d81b8b070d1d0482b6e68/config.yaml")
        
    def filter(self, audio, sample_rate = 16000):
        # audio = 'trials/Male speech-01989.wav'
        # pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        
        # print(type(audio[0]), audio.shape)
        audio = torch.from_numpy(np.float32(audio))
        # print(audio[0], audio.shape)
        audio = torch.reshape(audio, (1,-1))
        audio_in_memory = {"waveform": audio, "sample_rate": sample_rate}
        output = self.pipeline(audio_in_memory)
        grouping = []
        for turn, _, speaker in output.itertracks(yield_label=True):
            grouping.append([turn.start, turn.end])
        vad = []
        group = 0
        for i in range(0, 10000):
            if len(grouping)==group:
                vad.append(0)
            elif i/1000<grouping[group][0]:
                vad.append(0)
            elif i/1000<grouping[group][1]:
                vad.append(1)
            else:
                group += 1
                vad.append(1)
        return np.array(vad)

class AMP():
    def filter(self, audio, sample_rate):
        # samplerate, data = wavfile.read(fname)
        maxVolume = 0.5
        active = 0
        for i in audio:
            if i > maxVolume:
                active = 1
                break
        return active

class VAD():
    def __init__(self):
        self.ste = STE()
        self.amp = AMP()
        self.pyan = PYAN()
        self.resolution = None
        self.sample_rate = None

    def vader(self, audio, PYAN_filter, STE_filter, AMP_filter, plot = False, threshold_ratio = 0.5, frame_voting = False):
        
        prediction = 0
        if frame_voting:
            vad = []
            for i in range(self.resolution):
                count = np.max(PYAN_filter[(i*len(PYAN_filter)//self.resolution): ((i+1)*len(PYAN_filter)//self.resolution)])
                if(count==1):
                    vad.append(1)
                    continue
                count = np.max(STE_filter[(i*len(STE_filter)//self.resolution): ((i+1)*len(STE_filter)//self.resolution)])
                # count += np.max(WEB_filter[(i*len(WEB_filter)//resolution): ((i+1)*len(WEB_filter)//resolution)])
                count += AMP_filter
                if count>=2:
                    vad.append(1)
                else:
                    vad.append(0)
            vad = np.array(vad)
            
            if(vad.sum()>= threshold_ratio*self.resolution):
                prediction = 1
        else:
            pyan_v = PYAN_filter.sum()>= 0.5*self.resolution
            ste_v = STE_filter.sum()>= 0.5*self.resolution
            amp_v = AMP_filter == 1
            if pyan_v or (amp_v or ste_v):
                prediction = 1
            
        if plot:
            # sample_rate, signal = scipy.io.wavfile.read(audio)
            amplitude = max(audio)
            plt.subplots(1, 1, figsize=(20, 10))
            
            plt.plot([i/self.sample_rate for i in range(len(audio))], audio, label="Signal")
            plt.plot([i*10/len(STE_filter) for i in range(len(STE_filter))], STE_filter*amplitude+amplitude*0.02, 
                color = 'c', label="STE VAD", alpha=0.7, linestyle='dashed')
            # plt.plot([i*10/len(WEB_filter) for i in range(len(WEB_filter))], WEB_filter*amplitude-amplitude*0.02, 
            #     color = 'm', label="WEB VAD", alpha=0.7, linestyle='dashed')
            plt.plot([i/1000 for i in range(self.resolution)], [AMP_filter*amplitude-amplitude*0.02]*self.resolution, 
                color = 'm', label="AMP VAD", alpha=0.7, linestyle='dashed')
            plt.plot([i*10/len(PYAN_filter) for i in range(len(PYAN_filter))], PYAN_filter*amplitude-amplitude*0.04, 
                color = 'orange', label="PYAN VAD", alpha=0.7, linestyle='dashed')
            if frame_voting:
                plt.plot([i*10/len(vad) for i in range(len(vad))], vad*amplitude, color = 'k', label="ENSEMBLE VAD", 
                    linewidth=3, alpha=0.7)
            
            plt.legend(loc='best')
            plt.title('Prediction: ' + str(prediction))
            plt.xlabel('time (s)')
            plt.savefig('TestImages/' + str(audio[90]) + '.png')
        
        return prediction

    def predict(self, audio, sample_rate, threshold = -20, plot = False):
        # stereo_audio = AudioSegment.from_file(audio, format="wav")
        # mono_audios = stereo_audio.set_channels(1)
        # mono_audios.export(audio, format="wav")
        
        # old_samplerate, old_audio = scipy.io.wavfile.read(audio)
        # duration = old_audio.shape[0] / old_samplerate
        # time_old  = np.linspace(0, duration, old_audio.shape[0])
        # time_new  = np.linspace(0, duration, int(old_audio.shape[0] * 16000 / old_samplerate))
        # interpolator = scipy.interpolate.interp1d(time_old, old_audio.T)
        # new_audio = interpolator(time_new).T
        
        # scipy.io.wavfile.write(audio, 16000, np.round(new_audio).astype(old_audio.dtype))
        
        STE_filter = self.ste.filter(audio, sample_rate, threshold)
        
        AMP_filter = self.amp.filter(audio, sample_rate)
        
        PYAN_filter = self.pyan.filter(audio, sample_rate)
        # print(len(STE_filter), len(WEB_filter), len(PYAN_filter))

        self.resolution = (audio.size // sample_rate)*1000
        self.sample_rate = sample_rate
        
        return self.vader(audio, PYAN_filter, STE_filter, AMP_filter, plot)