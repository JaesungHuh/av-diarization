#!/usr/bin/python
# -*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import os
import math
import glob
import subprocess
import logging
from shutil import rmtree
from typing import Optional, Union, Tuple, List

import cv2
import python_speech_features

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn.functional as F

from .models.syncnetmodel import SyncNetModel
from .utils import load_checkpoint


# ==================== Get OFFSET ====================

def calc_pdist(feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 10) -> List[torch.Tensor]:
    win_size = vshift * 2 + 1
    feat2p = F.pad(feat2, (0, 0, vshift, vshift))
    dists = []

    for i in range(len(feat1)):
        dists.append(1 - F.cosine_similarity(feat1[[i], :].repeat(win_size, 1), feat2p[i:i+win_size, :]))
    return dists

# ==================== MAIN DEF ====================

class SyncNet(torch.nn.Module):
    def __init__(self, 
                 cache_dir : str, 
                 ckpt_dir : Optional[str] = None, 
                 device: Union[str, torch.device] = torch.device('cpu'), 
                 batch_size: int = 20):
        super(SyncNet, self).__init__()

        self.cache_dir = cache_dir
        self.crop_dir = os.path.join(cache_dir, 'pycrop')
        self.cropwav_dir = os.path.join(cache_dir, 'pycrop_wav')

        os.makedirs(self.cropwav_dir, exist_ok=True)

        self.model = SyncNetModel()
        self.model.to(device)
        self.model.load_state_dict(load_checkpoint("syncnet", download_root = ckpt_dir, device = device))
        self.model.eval()

        self.batch_size = batch_size
        self.vshift = 10
        self.device = device

    def evaluate(self, videofile: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ========== ==========
        # Convert files
        # ========== ==========
        temp_dir = os.path.join(self.cache_dir, 'pytmp')
        os.makedirs(temp_dir, exist_ok=True)

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", videofile,
            "-threads", "1", "-f", "image2", os.path.join(temp_dir, "%06d.jpg")
        ]
        subprocess.call(command, shell=False, stdout=None)

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", videofile,
            "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            os.path.join(temp_dir, 'audio.wav')
        ]
        subprocess.call(command, shell=False, stdout=None)

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        flist = glob.glob(os.path.join(temp_dir, '*.jpg'))
        flist.sort()

        for fname in flist:
            images.append(cv2.imread(fname))

        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.from_numpy(im.astype(float)).float()

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(temp_dir, 'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])

        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.from_numpy(cc.astype(float)).float()

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio)) / 16000) != (float(len(images)) / 25):
            logging.warning("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (float(len(audio)) / 16000, float(len(images)) / 25))

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        with torch.no_grad():
            for i in range(0, lastframe, self.batch_size):
                im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + self.batch_size))]
                im_in = torch.cat(im_batch, 0)
                im_out = self.model.forward_lip(im_in.to(self.device))
                im_feat.append(im_out.data.cpu())

                cc_batch = [cct[:, :, :, vframe*4:vframe*4+20] for vframe in range(i, min(lastframe, i + self.batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                cc_out = self.model.forward_aud(cc_in.to(self.device))
                cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========
        dists = calc_pdist(im_feat, cc_feat, vshift=self.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = self.vshift - minidx
        conf = torch.median(mdist) - minval

        fdist = np.stack([dist[minidx].numpy() for dist in dists])
        # fconf = torch.median(mdist).numpy() - fdist

        dists_npy = np.array([dist.numpy() for dist in dists])

        # Remove temp directory
        rmtree(temp_dir)

        return offset.numpy(), conf.numpy(), dists_npy

    def run(self) -> List[np.ndarray]:
        logging.info("Running syncnet...")
        flist = glob.glob(os.path.join(self.crop_dir, '0*.avi'))
        flist.sort()

        offsets = {}
        dists = []

        for idx, fname in enumerate(flist):
            offset, conf, dist = self.evaluate(videofile=fname)
            dists.append(dist)

            diagval = []
            numfr = dist.shape[0]
            for sidx, shift in enumerate(range(-self.vshift, self.vshift + 1)):
                diagval.append(dist[max(-shift, 0):min(numfr-shift, numfr), sidx])

            avgdiagval = np.array([np.mean(x) for x in diagval])
            diagoffset = self.vshift - np.argmin(avgdiagval)

            offsets[os.path.basename(os.path.splitext(flist[idx])[0])] = diagoffset

            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", flist[idx],
                "-async", "1", "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                flist[idx].replace(self.crop_dir, self.cropwav_dir).replace('.avi', '.wav')
            ]
            subprocess.call(command, shell=False, stdout=None)

        return dists


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    videofile = '/users/jaesung/voxconverse_method/data/original/sample.mp4'
    ckpt_dir = '/users/jaesung/voxconverse_method/data/syncnet_v2.model'

    syncnet = SyncNet(cache_dir=cache_dir, ckpt_dir=ckpt_dir, device=torch.device('cuda'))
    
    # If processing a single video, use evaluate:
    offset, conf, dists = syncnet.evaluate(videofile)
    
    # Alternatively, if you meant to process multiple cropped files, simply call run() with no arguments:
    # syncnet.run()