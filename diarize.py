#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch

from voxconverse.avdiarizer import AVDiarizer


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialze diarizer
    diarizer = AVDiarizer(args)

    # Run the diarizer
    diarizer.run(args.input, args.out_dir, device, args.cache_dir, args.visualize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="input video file", default="sample/sample.mp4")
    parser.add_argument("-o","--out_dir", type=Path, help="output directory", default="output")
    parser.add_argument("--cache_dir", type=Path, help="cache directory to store intermediate results", default=None)
    parser.add_argument("--ckpt_dir", type=Path, help="model directory", default=None)
    parser.add_argument("--visualize", action="store_true", help="visualize the results")

    parser.add_argument("--vad", type=str, help="vad", default="pywebrtcvad", choices=["pywebrtcvad", "silero"])
    parser.add_argument("--speaker_model", type=str, help="speaker model", default="resnetse34", choices=["ecapa-tdnn", "resnetse34"])
    args = parser.parse_args()
    main(args)


