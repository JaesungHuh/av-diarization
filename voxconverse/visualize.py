#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for VoxConverse method.
This script generates a video with visualizations of face detections.
"""

import glob
import os
import pickle
import subprocess

from typing import List, Dict

import cv2
import numpy as np
import tqdm
from scipy import signal
from scipy.io import wavfile


class Visualizer:
    """Visualization utility class for generating annotated video output."""
    
    def __init__(self, cache_dir: str, max_width: int = 720, frame_rate: int = 25):
        """
        Initialize the Visualizer.

        Parameters:
        cache_dir (str): The cache directory path.
        max_width (int): Maximum width for output video.
        frame_rate (int): Frame rate for the output video.
        """
        self.cache_dir = cache_dir
        self.frames_dir = os.path.join(cache_dir, 'pyframes')
        self.avi_dir = os.path.join(cache_dir, 'pyavi')
        self.max_width = max_width
        self.frame_rate = frame_rate

    def add_detections_to_video(self, 
                                flist: List[str], 
                                faces: List[List[Dict]], 
                                vonly_file: str, 
                                out_file: str) -> None:
        """
        Overlay face detections on frames and create a video file.

        Parameters:
        flist (List[str]): List of frame image file paths.
        faces (List[List[Dict]]): List of lists with face detection dictionaries for each frame.
        vonly_file (str): Path to the temporary video file (without audio).
        out_file (str): Final output video file path.
        """
        first_image = cv2.imread(flist[0])
        fw, fh = first_image.shape[1], first_image.shape[0]

        if fw > self.max_width:
            sf = float(self.max_width) / float(fw)
            fw, fh = int(sf * fw), int(sf * fh)
        else:
            sf = 1

        fhb = int(sf * fh / 40)
        # Removed unused variable 'fts'
        # fts = float(fhb) / 60

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut = cv2.VideoWriter(vonly_file, fourcc, self.frame_rate, (fw, fh))

        for fidx, fname in tqdm.tqdm(enumerate(flist), desc=f"Writing to {out_file}", total=len(flist)):
            image = cv2.imread(fname)

            if sf != 1:
                image = cv2.resize(image, (fw, fh))

            for face in faces[fidx]:
                face['x'] *= sf
                face['y'] *= sf
                face['s'] *= sf

                clr3 = int(max(min(255, face['conf'] * 1000), 0))

                cv2.rectangle(
                    image,
                    (int(face['x'] - face['s']), int(face['y'] - face['s'])),
                    (int(face['x'] + face['s']), int(face['y'] + face['s'])),
                    (0, clr3, 255 - clr3),
                    3
                )
                cv2.putText(
                    image,
                    'TR %d, ID %d' % (face['track'], face['identity']),
                    (int(face['x'] - face['s']), int(face['y'] - face['s'])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )
            vOut.write(image)
        vOut.release()

    def smooth_faces(self, 
                     num_frames: int, 
                     tracks: List[Dict], 
                     dists: List[np.ndarray], 
                     face_id: List[int]) -> List:
        """
        Smooth face detection confidences over frames.

        Parameters:
        num_frames (int): Total number of frames.
        tracks (List[Dict]): List of tracking dictionaries.
        dists (List[np.ndarray]): List of distance arrays from SyncNet per track.
        face_id (List[int]): List of face identities per track.

        Returns:
        List: A list (one per frame) with face detection dictionaries.
        """
        faces = [[] for _ in range(num_frames)]

        for tidx, track in enumerate(tracks):
            try:
                mean_dists = np.mean(np.stack(dists[tidx], 1), 1)
                minidx = np.argmin(mean_dists, 0)
                # Removed unused variable 'minval'
                # minval = mean_dists[minidx]

                fdist = np.stack([dist[minidx] for dist in dists[tidx]])
                fdist = np.pad(fdist, (3, 3), 'constant', constant_values=10)

                fconf = np.median(mean_dists) - fdist
                fconfm = signal.medfilt(fconf, kernel_size=13)

                for fidx, frame in enumerate(track['track']['frame'].tolist()):
                    faces[frame].append({
                        'track': tidx,
                        'identity': face_id[tidx],
                        'conf': fconfm[fidx],
                        's': track['proc_track']['s'][fidx],
                        'x': track['proc_track']['x'][fidx],
                        'y': track['proc_track']['y'][fidx]
                    })
            except (IndexError, KeyError, ValueError):
                # Log the exception here if needed.
                continue
        return faces

    def run(self, 
            tracks: List[Dict], 
            dists: List[np.ndarray], 
            face_id: List[int], 
            out_file: str = 'out_vis.mp4') -> None:
        """
        Generate a visualization video by combining frames with face detections and audio.

        Parameters:
        tracks (List[Dict]): List of tracking dictionaries.
        dists (List[np.ndarray]): List of distance arrays per track.
        face_id (List[int]): List of face identities per track.
        out_file (str): The final output video file path.
        """
        flist = glob.glob(os.path.join(self.frames_dir, '*.jpg'))
        flist.sort()
        num_frames = len(flist)

        # Removed unused computation related to audio power.
        audio_file = os.path.join(self.avi_dir, 'audio.wav')
        _sample_rate, audio = wavfile.read(audio_file)

        faces = self.smooth_faces(num_frames, tracks, dists, face_id)
        vonly_file = os.path.join(self.avi_dir, 'vonly_file.avi')
        self.add_detections_to_video(flist, faces, vonly_file, out_file)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", vonly_file,
            "-i", audio_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "-2",
            out_file
        ]

        subprocess.call(command, shell=False, stdout=None)


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    with open('/users/jaesung/voxconverse_method/temp/pywork/tracks.pckl', 'rb') as f:
        tracks = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/activesd.pckl', 'rb') as f:
        dists = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/faceidx.pkl', 'rb') as f:
        face_id = pickle.load(f)

    visualizer = Visualizer(cache_dir)
    visualizer.run(tracks, dists, face_id, out_file='out_vis.mp4')