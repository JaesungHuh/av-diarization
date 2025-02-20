#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
import tempfile
import argparse

from typing import Optional

from .preprocessor import Preprocessor
from .syncnet import SyncNet
from .facecluster import FaceCluster
from .vad import Vad
from .speakernet import SpeakerNet
from .diarizer import Diarizer
from .visualize import Visualizer

logging.getLogger("exifread").setLevel(level=logging.CRITICAL)

class AVDiarizer():
    def __init__(self, 
                 args: argparse.Namespace):
        self.args = args
    
    def run(self, 
            in_file: str, 
            out_dir: str = "output", 
            device: str = "cpu", 
            cache_dir: Optional[str] = None, 
            visualize: bool = False):
        """
        Run the AVDiarizer

        Args:
            in_file: str, input video file
            out_dir: str, output directory
            device: str, torch.device to use
            cache_dir: str, cache directory
            visualize: bool, whether to visualize the ASD results
        """

        ckpt_dir = self.args.ckpt_dir
        # This is the main function that will be called by the CLI
        temp_dir = None

        if cache_dir is None:
            # Create the temporary directory and remove this at the end of the 
            temp_dir = tempfile.TemporaryDirectory()
            cache_dir = temp_dir.name
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        
        # Preprocess
        preprocessor = Preprocessor(cache_dir, ckpt_dir, device)
        tracks = preprocessor.run(in_file)

        # Syncnet
        syncnet = SyncNet(cache_dir, ckpt_dir, device)
        dists = syncnet.run()

        # Face cluster
        face_clusterer = FaceCluster(cache_dir, ckpt_dir, device)
        face_ids = face_clusterer.run(tracks)

        # Vad
        vad = Vad(cache_dir, ckpt_dir, self.args.vad, device)
        vadres = vad.run()

        # Speaker embedding extractor
        spknet = SpeakerNet(cache_dir, ckpt_dir, self.args.speaker_model, device)
        spkfeats = spknet.run()

        # Diarizer using the results from the previous steps
        diarizer = Diarizer(cache_dir, self.args.out_dir)
        result = diarizer.run(tracks, dists, vadres, face_ids, spkfeats, in_file)  # result contains list of [(start, end, spk)]

        if visualize:
            out_file = os.path.join(out_dir, 'out.mp4')
            visualizer = Visualizer(cache_dir = cache_dir)
            visualizer.run(tracks, dists, face_ids, out_file)

        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ ==  '__main__':
    pass