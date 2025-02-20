#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for performing Voice Activity Detection (VAD) using different models.
"""

import os
import glob
import contextlib
import wave
import pickle
import logging
from typing import Dict, List, Optional

import webrtcvad
import torch
import torchaudio

from .utils import get_speech_timestamps, vad_collector, frame_generator

# Constants for magic numbers for pywebrtcvad
SAMPLE_DURATION_MS = 30
MAX_SILENCE_MS = 300
SEGMENT_FACTOR = 32000


class Vad:
    """
    Class for Voice Activity Detection (VAD) using either pywebrtcvad or Silero-VAD.
    """

    def __init__(self, 
                 cache_dir: str, 
                 ckpt_dir: Optional[str] = None,
                 vad_model: str = 'pywebrtcvad',
                 device: torch.device = torch.device('cpu')):
        """
        Initialize the Vad instance.

        Args:
            cache_dir (str): Directory for cache.
            ckpt_dir (Optional[str]): Directory for checkpoint (if any).
            vad_model (str): VAD model to use ('pywebrtcvad' or 'silero').
        """
        self.cache_dir = cache_dir
        self.ckpt_dir = ckpt_dir
        self.vad_model = vad_model
        self.device = device

        self.avi_dir = os.path.join(cache_dir, 'pyavi')
        self.cropwav_dir = os.path.join(cache_dir, 'pycrop_wav')
        self.sampling_rate = 16000

        if not os.path.isdir(self.cropwav_dir):
            raise FileNotFoundError(f'No files in {self.cropwav_dir}')
        if vad_model not in ['pywebrtcvad', 'silero']:
            raise ValueError(f'VAD model {vad_model} not supported')

    def run_webrtc(self, aggressiveness: int, filename: str) -> List:
        """
        Run VAD using pywebrtcvad.

        Args:
            aggressiveness (int): The aggressiveness mode for VAD.
            filename (str): Path to the WAV file.

        Returns:
            List: A list of segments, each represented as [start_time, duration].
        """
        with contextlib.closing(wave.open(filename, 'rb')) as wf:
            sample_rate = wf.getframerate()
            if sample_rate != self.sampling_rate:
                raise ValueError('Sample rate must be 16000')
            audio = wf.readframes(wf.getnframes())

        vad = webrtcvad.Vad(aggressiveness)
        frames = list(frame_generator(SAMPLE_DURATION_MS, audio, sample_rate))
        segments = vad_collector(sample_rate, SAMPLE_DURATION_MS, MAX_SILENCE_MS, vad, frames)

        segs = []
        for timestamp, segment in segments:
            segs.append([timestamp, float(len(segment)) / SEGMENT_FACTOR])

        return segs

    def run_silero(self, filename: str, device: torch.device = torch.device('cpu')) -> List:
        """
        Run VAD using silero-vad.

        Args:
            filename (str): Path to the WAV file.

        Returns:
            List: A list of segments, each represented as [start_time, duration].
        """
        torch.set_num_threads(1)
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', verbose=False)
        model.to(self.device)

        wav, sr = torchaudio.load(filename, backend='soundfile')

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            wav = transform(wav)
            sr = self.sampling_rate

        wav = wav.squeeze(0).to(device)
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

        segs = []
        for seg in speech_timestamps:
            segs.append([seg['start'], seg['end'] - seg['start']])

        return segs

    def run(self) -> Dict:
        """
        Run VAD for all files in the cropwav_dir.

        Returns:
            Dict: A dictionary with file names as keys and VAD results as values.
        """
        logging.info("Running VAD...")
        files = glob.glob(os.path.join(self.cropwav_dir, '0*.wav'))

        if not files:
            logging.warning(f'No files in {self.cropwav_dir} which might cause degradation of this pipeline')

        files.append(os.path.join(self.avi_dir, 'audio.wav'))

        vadres = {}

        if self.vad_model == 'pywebrtcvad':
            logging.info("Running Pywebrtcvad...")
            for fname in files:
                vadres[os.path.basename(fname)] = self.run_webrtc(aggressiveness=3, filename=fname)
        else:  # silero
            logging.info("Running Silero-VAD...")
            for fname in files:
                vadres[os.path.basename(fname)] = self.run_silero(filename=fname, device=self.device)

        return vadres


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'

    vad = Vad(cache_dir=cache_dir, ckpt_dir=None, vad_model='pywebrtcvad')
    vadres = vad.run()

    with open(os.path.join(cache_dir, 'pywork', 'webrtc.pkl'), 'wb') as f:
        pickle.dump(vadres, f)