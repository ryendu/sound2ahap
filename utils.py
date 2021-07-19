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
