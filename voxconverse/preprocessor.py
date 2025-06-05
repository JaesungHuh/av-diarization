#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import tqdm
from scipy import signal
from scipy.interpolate import interp1d

from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager

from .models.s3fd import S3FD
from .utils import load_checkpoint


class Preprocessor:
    def __init__(self, 
                 cache_dir: str,
                 ckpt_dir: Optional[str] = None,
                 device: Union[str, torch.device] = "cpu",
                 frame_rate: int = 25,
                 crop_scale: float = 0.40,
                 min_track: int = 50,
                 num_failed_det: int = 25,
                 min_face_size: int = 40,
                 facedet_scale: float = 0.35):
        
        self.cache_dir = cache_dir
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.frame_rate = frame_rate
        self.crop_scale = crop_scale
        self.min_track = min_track
        self.num_failed_det = num_failed_det
        self.min_face_size = min_face_size
        self.facedet_scale = facedet_scale

        self._setup_paths()
    
    def _setup_paths(self) -> None:
        """
        Setup the paths for the preprocessor
        """
        self.avi_dir = os.path.join(self.cache_dir, 'pyavi')
        self.crop_dir = os.path.join(self.cache_dir, 'pycrop')
        self.frames_dir = os.path.join(self.cache_dir, 'pyframes')

        os.makedirs(self.avi_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

    @staticmethod
    def bb_intersection_over_union(boxA: List[float], boxB: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes

        Args:
            boxA: List[float], first bounding box
            boxB: List[float], second bounding box

        Returns:
            float, IoU between the two bounding boxes
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
      
        interArea = max(0, xB - xA) * max(0, yB - yA)
      
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
      
        iou = interArea / float(boxAArea + boxBArea - interArea)
      
        return iou
    
    def track_shot(self, scenefaces: List[List[Dict]]) -> List[Dict]:
        """
        Track faces per each shot

        Args:
            scenefaces: List[List[Dict]], faces per each shot

        Returns:
            List[Dict], tracks
        """

        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []

        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if not track:
                        track.append(face)
                        framefaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break

            if not track:
                break
            elif len(track) > self.min_track:
                framenum = np.array([f['frame'] for f in track])
                bboxes = np.array([np.array(f['bbox']) for f in track])

                frame_i = np.arange(framenum[0], framenum[-1] + 1)

                bboxes_i = []
                for ij in range(4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)

                if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), 
                       np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > self.min_face_size:
                    tracks.append({'frame': frame_i, 'bbox': bboxes_i})

        return tracks

    def crop_video(self, track: Dict, cropfile: str) -> Dict:
        """
        Crop the video based on face detection results per each track

        Args:
            track: Dict, track
            cropfile: str, crop file

        Returns:
            Dict, track
        """

        flist = glob.glob(os.path.join(self.frames_dir, '*.jpg'))
        flist.sort()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, self.frame_rate, (224, 224))

        dets = {'x': [], 'y': [], 's': []}

        for det in track['bbox']:
            dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2) 
            dets['y'].append((det[1] + det[3]) / 2)  # crop center x 
            dets['x'].append((det[0] + det[2]) / 2)  # crop center y

        # Smooth detections
        dets['s'] = signal.medfilt(dets['s'], kernel_size=13)   
        dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

        for fidx, frame in enumerate(track['frame']):
            cs = self.crop_scale

            bs = dets['s'][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 

            image = cv2.imread(flist[frame])
            
            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 
                           'constant', constant_values=(110, 110))
            my = dets['y'][fidx] + bsi  # BBox center Y
            mx = dets['x'][fidx] + bsi  # BBox center X

            face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)),
                         int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
            
            vOut.write(cv2.resize(face, (224, 224)))

        audiotmp = tempfile.NamedTemporaryFile(suffix='.wav')
        audiotmp = audiotmp.name
        audiostart = track['frame'][0] / self.frame_rate
        audioend = (track['frame'][-1] + 1) / self.frame_rate

        vOut.release()
        
        command = [
            "ffmpeg", 
            "-hide_banner", 
            "-loglevel", "error",
            "-y",
            "-i", os.path.join(self.avi_dir, 'audio.wav'),
            "-ss", str(audiostart),
            "-to", str(audioend),
            audiotmp
        ]
        subprocess.call(command, shell=False, stdout=None)

        command = [
            "ffmpeg", 
            "-hide_banner", 
            "-loglevel", "error",
            "-y",
            "-i", f"{cropfile}t.avi",
            "-i", audiotmp,
            "-c:v", "copy",
            "-c:a", "copy",
            f"{cropfile}.avi"
        ]
        subprocess.call(command, shell=False, stdout=None)

        os.remove(cropfile + 't.avi')

        return {'track': track, 'proc_track': dets}

    def inference_video(self) -> List[Dict]:
        """
        Detect faces in the video using S3FD

        Returns:
            List[Dict], faces
        """
        DET = S3FD(device=self.device)
        DET.net.load_state_dict(load_checkpoint("s3fd", self.ckpt_dir, self.device))
        DET.net.to(self.device)
        DET.net.eval()

        flist = glob.glob(os.path.join(self.frames_dir, '*.jpg'))
        flist.sort()

        dets = []
            
        for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist), desc="Detecting faces"):
            image = cv2.imread(fname)

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[self.facedet_scale])

            dets.append([])
            for bbox in bboxes:
                dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})

        return dets

    def scene_detect(self) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
        Detect scenes in the video using scenedetect

        Returns:
            List[Tuple[FrameTimecode, FrameTimecode]], scenes
        """
        video_manager = VideoManager([os.path.join(self.avi_dir, 'video.avi')])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)

        # Add ContentDetector algorithm (constructor takes detector options like threshold).
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()

        video_manager.set_downscale_factor()

        video_manager.start()

        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list(base_timecode)

        if not scene_list:
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

        logging.info(f'{os.path.join(self.avi_dir, "video.avi")} - scenes detected {len(scene_list)}')

        return scene_list

    def _process_video(self) -> None:
        """
        Process the video, extract the frames and audio
        """
        logging.info("Processing video...")
        # Convert video
        command = [
            "ffmpeg", 
            "-hide_banner", 
            "-loglevel", "error",
            "-y", 
            "-i", self.videofile, 
            "-async", "1", 
            "-qscale:v", "2", 
            "-r", "25", 
            os.path.join(self.avi_dir, 'video.avi')
        ]
        subprocess.call(command, shell=False, stdout=None)

        # Extract frames
        logging.info("Extracting video frames...")
        command = [
            "ffmpeg", 
            "-hide_banner", 
            "-loglevel", "error",
            "-y", 
            "-i", os.path.join(self.avi_dir, 'video.avi'), 
            "-threads", "1", 
            "-qscale:v", "2", 
            "-f", "image2", 
            os.path.join(self.frames_dir, '%06d.jpg')
        ]
        subprocess.call(command, shell=False, stdout=None)

        # Extract audio
        logging.info("Extracting audio from video...")
        command = [
            "ffmpeg", 
            "-hide_banner", 
            "-loglevel", "error",
            "-y", 
            "-i", os.path.join(self.avi_dir, 'video.avi'),
            "-ac", "1", 
            "-vn", 
            "-acodec", "pcm_s16le", 
            "-ar", "16000",
            os.path.join(self.avi_dir, 'audio.wav')
        ]
        subprocess.call(command, shell=False, stdout=None)

    def run(self, videofile: str) -> List:
        """
        Run the preprocessor results in face tracks

        Args:
            videofile: str, input video file

        Returns:
            List, tracks
        """
        logging.info(f"Running preprocessor with {videofile}")

        if not os.path.exists(videofile):
            print(f'Video file : {videofile} not found')
            return

        self.videofile = videofile

        # Create directories and process
        self._process_video()

        # Run detection and tracking
        faces = self.inference_video()
        scene = self.scene_detect()

        # Track faces
        alltracks = []
        vidtracks = []

        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= self.min_track:
                alltracks.extend(self.track_shot(faces[shot[0].frame_num:shot[1].frame_num]))

        logging.info(f"Detected {len(alltracks)} tracks")

        # Process tracks
        for ii, track in enumerate(alltracks):
            vidtracks.append(self.crop_video(track, os.path.join(self.crop_dir, f'{ii:05d}')))

        return vidtracks


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/cache'
    videofile = '/users/jaesung/voxconverse_method/sample/sample.mp4'

    preprocessor = Preprocessor(cache_dir)
    vidtracks = preprocessor.run(videofile)