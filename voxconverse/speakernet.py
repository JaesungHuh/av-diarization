#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import logging
from typing import Optional

from speechbrain.inference.speaker import EncoderClassifier
import torch
import torch.nn as nn
import torchaudio

from .models.resnetse34v2 import ResNetSE34V2
from .utils import load_checkpoint


class SpeakerNet(nn.Module):
    def __init__(self, 
                 cache_dir: str, 
                 ckpt_dir : Optional[str] = None, 
                 model_type: str = 'ecapa-tdnn', 
                 device: torch.device = torch.device('cpu'),
                 max_frames: int = 200):
        super(SpeakerNet, self).__init__()
        self.cache_dir = cache_dir
        self.ckpt_dir = ckpt_dir
        self.work_dir = os.path.join(cache_dir, 'pywork')
        self.avi_dir = os.path.join(cache_dir, 'pyavi')
        self.device = device
        self.model_type = model_type

        assert model_type in ['ecapa-tdnn', 'resnetse34'], \
            f'Model type {model_type} not supported'
        if self.model_type == 'ecapa-tdnn':
            hyperparameters = {'device': self.device.type}
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=self.ckpt_dir,
                run_opts=hyperparameters
            )
        else:
            self.model = ResNetSE34V2(
                nOut=512, encoder_type="ASP", n_mels=64, log_input=True
            )
            self.model.load_state_dict(
                load_checkpoint("speakernet", download_root = self.ckpt_dir, device = device)
            )
            self.model.to(self.device)
        self.model.eval()
        self.max_frames = max_frames
    
    @torch.no_grad()
    def run_resnetse34(self, fname: str) -> torch.Tensor:
        inp1, fs = torchaudio.load(fname)
        feats = []
        for ii in range(0, inp1.size()[-1] - self.max_frames * 160, 3200):
            feats.append(
                self.model.forward(
                    inp1[:, ii:ii + self.max_frames * 160].cuda()
                ).detach().cpu()
            )
        return feats

    @torch.no_grad()
    def run_ecapa(self, fname: str) -> torch.Tensor:
        inp1, fs = torchaudio.load(fname)
        feats = []
        for ii in range(0, inp1.size()[-1] - self.max_frames * 160, 3200):
            x = self.model.encode_batch(
                inp1[:, ii:ii + self.max_frames * 160].cuda()
            ).detach().cpu()
            feats.append(torch.squeeze(x, dim=1))
        return feats

    @torch.no_grad()
    def run(self) -> torch.Tensor:
        """
        Run speaker embedding extraction

        Returns:
            torch.Tensor, speaker features
        """
        logging.info("Running speaker embedding extraction...")
        filename = os.path.join(self.avi_dir, 'audio.wav')

        if self.model_type == 'ecapa-tdnn':
            logging.info("Running ECAPA-TDNN...")
            feats = self.run_ecapa(filename)
        else:
            logging.info("Running ResNetSE34...")
            feats = self.run_resnetse34(filename)
        feats = torch.cat(feats, dim=0)
        return feats


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    model_type = 'resnetse34'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    speakernet = SpeakerNet(cache_dir=cache_dir, model_type=model_type, device=device)
    feats = speakernet.run()
    torch.save(feats, os.path.join(cache_dir, 'pywork', 'resnet.pt'))