import sklearn
import scipy
import torch
from torch import nn
from torch import functional as F
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
from scipy import signal
import pyAudioAnalysis
from pyAudioAnalysis import ShortTermFeatures
import datetime
import os
import json
import librosa
import librosa.display


def normalize(x):
    multiplier = (15*0.005**x+1)
    res = x*multiplier
    if res < 0.05:
        res = res * 6
    elif res < 0.1:
        res = res * 4
    elif res < 0.2:
        res = res * 2.5
    return res


def RMSE(data):  # source - https://rramnauth2220.github.io/blog/posts/code/200525-feature-extraction.html
    hop_length = 256
    frame_length = 512

    # compute sum of signal square by frame
    energy = np.array([
        sum(abs(data[i:i+frame_length]**2))
        for i in range(0, len(data), hop_length)
    ])
    energy.shape

    # compute RMSE over frames
    rmse = librosa.feature.rms(
        data, frame_length=frame_length, hop_length=hop_length, center=True)
    rmse.shape
    rmse = rmse[0]
    return rmse

# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb


def mel_sync(M, beats, plot=False):
    # feature.sync will summarize each beat event by the mean feature vector within that beat
    M_sync = librosa.util.sync(M, beats)
    if plot:
        plt.figure(figsize=(12, 6))
        # Let's plot the original and beat-synchronous features against each other
        plt.subplot(2, 1, 1)
        librosa.display.specshow(M)
        plt.title('MFCC-$\Delta$-$\Delta^2$')
        plt.yticks(np.arange(0, M.shape[0], 13), [
                   'MFCC', '$\Delta$', '$\Delta^2$'])
        plt.colorbar()
        plt.subplot(2, 1, 2)
        librosa.display.specshow(M_sync, x_axis='time',
                                 x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))
        plt.yticks(np.arange(0, M_sync.shape[0], 13), [
                   'MFCC', '$\Delta$', '$\Delta^2$'])
        plt.title('Beat-synchronous MFCC-$\Delta$-$\Delta^2$')
        plt.colorbar()
        plt.tight_layout()
    return M_sync


# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb
def beat_sync_chroma(C, beats, samplerate, plot=False):
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            C_harmonic, sr=samplerate, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time')
        plt.title('Chroma')
        plt.colorbar()
        plt.subplot(2, 1, 2)
        librosa.display.specshow(C_sync, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time',
                                 x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))
        plt.title('Beat-synchronous Chroma (median aggregation)')
        plt.colorbar()
        plt.tight_layout()
    return C_sync

# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb


def melspect(data, samplerate, plot=False, title='mel power spectrogram'):
    S = librosa.feature.melspectrogram(data, sr=samplerate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    if plot:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=samplerate,
                                 x_axis='time', y_axis='mel')
        plt.title(title)
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
    return S


# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb
def track_beats(precussive, samplerate, plot=False, log_S=None):
    tempo, beats = librosa.beat.beat_track(
        y=precussive, sr=samplerate, units="time")
    if plot:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=samplerate,
                                 x_axis='time', y_axis='mel')
        # Let's draw transparent lines over the beat frames
        plt.vlines(librosa.frames_to_time(beats),
                   1, 0.5 * samplerate,
                   colors='w', linestyles='-', linewidth=2, alpha=0.5)
        plt.axis('tight')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()
    return tempo, beats


# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb
def chromagram(data, samplerate, plot=True, title="Chromagram"):
    C = librosa.feature.chroma_cqt(y=data, sr=samplerate, bins_per_octave=36)
    if plot:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            C, sr=samplerate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
    return C


# source: https://github.com/librosa/librosa/blob/main/examples/LibROSA%20demo.ipynb
def MFCC(S, samplerate, plot=False):
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S, ref=np.max), n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(mfcc)
        plt.ylabel('MFCC')
        plt.colorbar()
        plt.subplot(3, 1, 2)
        librosa.display.specshow(delta_mfcc)
        plt.ylabel('MFCC-$\Delta$')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(delta2_mfcc, sr=samplerate, x_axis='time')
        plt.ylabel('MFCC-$\Delta^2$')
        plt.colorbar()
        plt.tight_layout()
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return M


def plot_bar(data, length):
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(40, 6))
    plt.bar(time, data, label="Left channel")
    plt.legend()
    plt.xlabel("Time [s]", )
    plt.xticks(np.arange(0, length+1, 3.0))
    plt.ylabel("Amplitude")
    plt.show()


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def plot_wav(data, length):
    time = np.linspace(0., length, data.shape[0])
    plt.figure(figsize=(40, 6))
    plt.plot(time, data[:, 0], label="Left channel")
    plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]", )
    plt.xticks(np.arange(0, length+1, 3.0))
    plt.ylabel("Amplitude")
    plt.show()


def factors_of(x):
    res = []
    for i in range(1, x + 1):
        if x % i == 0:
            res.append(i)
    return res


def closest_number_in_list(val, list_):
    return min(list_, key=lambda x: abs(x-val))


class AHAP:
    def __init__(self):
        self.data = {
            "Version": 1.0,
            "Metadata": {
                "Project": "Basis",
                "Created": str(datetime.datetime.now()),
                "Description": "AHAP file generated based on a audio file by Basis app.",
                "Created By": "Ryan Du"
            },
            "Pattern": []
        }

    def add_event(self, etype, time, parameters, event_duration=None, event_waveform_path=None):
        """
        Adds an event to the pattern
        etype: type of event
            - possible values: AudioContinuous, AudioCustom, HapticTransient, and HapticContinuous
        time: time of event
            - in seconds
        parameters: event parameters
            - as a list of dictionaries
        """
        pattern = {
            "Event": {
                "Time": time,
                "EventType": etype,
                "EventParameters": parameters
            }
        }
        if event_duration != None:
            pattern["Event"]["EventDuration"] = event_duration
        if event_waveform_path != None:
            pattern["Event"]["EventWaveformPath"] = event_waveform_path
        self.data["Pattern"].append(pattern)

    def add_haptic_transient_event(self, time, haptic_intensity=0.5, haptic_sharpness=0.5):
        """
        Adds a haptic transient event to the pattern
        time: time of event
            - in seconds
        haptic_intensity: intensity of haptic
        haptic_sharpness: sharpness of haptic
        """
        parameters = [
            {
                "ParameterID": "HapticIntensity",
                "ParameterValue": haptic_intensity,
            },
            {
                "ParameterID": "HapticSharpness",
                "ParameterValue": haptic_sharpness,
            }
        ]

        self.add_event(etype="HapticTransient",
                       time=time, parameters=parameters)

    def add_haptic_continuous_event(self, time, event_duration=1, haptic_intensity=0.5, haptic_sharpness=0.5):
        """
        Adds a haptic continuous event to the pattern
        time: time of event
            - in seconds
        haptic_intensity: intensity of haptic
        haptic_sharpness: sharpness of haptic
        """
        parameters = [
            {
                "ParameterID": "HapticIntensity",
                "ParameterValue": haptic_intensity,
            },
            {
                "ParameterID": "HapticSharpness",
                "ParameterValue": haptic_sharpness,
            }
        ]

        self.add_event(etype="HapticContinuous", time=time,
                       parameters=parameters, event_duration=event_duration)

    def add_audio_custom_event(self, time, wav_filepath, volume=0.75):
        """
        Adds an audio custom event to the pattern
        time: time of event
            - in seconds
        wav_filepath: path to the wav file containing the sound
        volume: volume from 0 to 1
        """
        parameters = [
            {
                "ParameterID": "AudioVolume",
                "ParameterValue": volume,
            }
        ]
        self.add_event(etype="AudioCustom", time=time,
                       parameters=parameters, event_waveform_path=wav_filepath)

    def add_parameter_curve(self, parameter_id, start_time, control_points):
        """
        Adds a parameter curve to the pattern
        parameter_id: the parameter to dynamically change
            - possible values: HapticIntensityControl, HapticSharpnessControl, HapticAttackTimeControl, HapticDecayTimeControl, HapticReleaseTimeControl, AudioBrightnessControl, AudioPanControl, AudioPitchControl, AudioVolumeControl, AudioAttackTimeControl, AudioDecayTimeControl, AudioReleaseTimeControl
        start_time: time of the start of the curve
            - in seconds
        control_points: list of control points
            - as a list of dictionaries in the format: [{"Time":time,"ParameterValue":value}]
        """
        pattern = {
            "ParameterCurve": {
                "ParameterID": parameter_id,
                "Time": start_time,
                "ParameterCurveControlPoints": control_points
            }
        }

        self.data["Pattern"].append(pattern)

    def print_data(self):
        print(self.data)

    def export(self, filename, path):
        with open(os.path.join(path, filename), 'w') as f:
            f.write(json.dumps(self.data))
