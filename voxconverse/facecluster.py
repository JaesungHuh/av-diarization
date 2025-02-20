#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from typing import Optional, List, Dict
import pickle
import logging
import glob

import cv2
from scipy.spatial.distance import cdist
import numpy as np

import torch
import torch.nn as nn

from .models.vggface import resnet50
from .utils import load_checkpoint


class FaceCluster(nn.Module):
    def __init__(self, 
                 cache_dir: str, 
                 ckpt_dir: Optional[str] = None, 
                 device: torch.device = torch.device('cpu')):
        super(FaceCluster, self).__init__()
        self.cache_dir = cache_dir
        self.device = device
        self.crop_dir = os.path.join(cache_dir, 'pycrop')

        assert os.path.isdir(self.crop_dir), f'No files in {self.crop_dir}'

        self.model = resnet50(num_classes=8631, include_top=False)
        self.model.to(device)
        self.model.load_state_dict(load_checkpoint("faceembedding", download_root = ckpt_dir, device = device))
        self.model.eval()

    @torch.no_grad()
    def extract(self, img_file: str) -> np.ndarray:
        """
        Extract face features from a img / video file

        Args:
            img_file: str, path to the image / video file

        Returns:
            np.ndarray, face features
        """
        cap = cv2.VideoCapture(img_file)
        total_frames = cap.get(7)
        getframes = (int(1 * total_frames / 5), int(2 * total_frames / 5), int(3 * total_frames / 5), int(4 * total_frames / 5))

        images = []

        if img_file[-3:] == 'jpg':
            img = cv2.imread(img_file,1)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img,axis=0)

        else:
            for ii in getframes:
                cap.set(1,ii)
                ret, image = cap.read()
                image = image[0:160,32:192,:]
                if ret:
                    images.append(cv2.resize(image, (224, 224)))
                    pastimage = image
                else:
                    images.append(cv2.resize(pastimage, (224, 224)))

            img = np.stack(images,0)
        
        img = np.transpose(img,(0,3,1,2))

        imgs = torch.FloatTensor(img).to(self.device)

        output = self.model(imgs)  # N C H W torch.Size([1, 1, 401, 600])

        output = output.view(output.size(0), -1)
        output = output.data.cpu().numpy()

        return output

    def run(self, tracks: List[Dict]) -> List[int]:
        """
        Run face clustering

        Args:
            tracks: List[Dict], list of tracks

        Returns:
            List[int], list of face ids
        """
        logging.info("Running face clustering with Face tracks...")

        ## list of cropped face tracks
        flist = glob.glob(os.path.join(self.crop_dir,'0*.avi'))
        flist.sort()

        ## average location of each face track
        vtlist = []
        for ii, track in enumerate(tracks):
            vtlist.append(track['track']['frame'])

        # sanity check - number of face tracks
        if len(vtlist) != len(flist):
            raise ValueError('Wrong number of tracks')

        ## ========== EXTRACT FEATURES ===========

        ## extract vid face feature
        feats = []
        for vid in flist:
            feat = self.extract(vid)
            feats.append(feat)

        faceinfo    = []
        faceidx     = []
        for idx, feat in enumerate(feats):
            nonoverlap = []
            for fidx, finfo in enumerate(faceinfo):
                if len(set(vtlist[idx]) & set(finfo['frame'])) == 0:
                    nonoverlap.append(fidx)
            
            if nonoverlap == []:
                faceinfo.append({'trackidx':[idx],'frame':vtlist[idx],'feat':feat})
                faceidx.append(len(faceinfo)-1)
                logging.info('New group %d as %d due to overlap'%(idx,len(faceinfo)-1))
            else:
                cosdists = []
                for x in nonoverlap:
                    x = cdist(feat, faceinfo[x]['feat'], metric='cosine')
                    cosdists.append(np.median(x))
                
                mval = np.min(np.array(cosdists))
                midx = nonoverlap[np.argmin(np.array(cosdists))]

                if mval <= 0.40:
                    # Assign to old group
                    faceinfo[midx]['frame'] = np.concatenate((faceinfo[midx]['frame'],vtlist[idx]),0)
                    faceinfo[midx]['feat'] = np.concatenate((faceinfo[midx]['feat'],feat),0)
                    faceinfo[midx]['trackidx'].append(idx)
                    faceidx.append(midx)
                    logging.info('Assigned %d to group %d with cos dist %.2f'%(idx,midx,mval))
                else:
                    # Assign to new group
                    faceinfo.append({'trackidx':[idx],'frame':vtlist[idx],'feat':feat})
                    faceidx.append(len(faceinfo)-1)
                    logging.info('Track %d as new group %d due to low feature distance with group %d (%.2f)'%(idx,len(faceinfo)-1,midx,mval))

        return faceidx


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    with open('/users/jaesung/voxconverse_method/temp/pywork/tracks.pckl', 'rb') as f:
        tracks = pickle.load(f)
    # print(tracks)
    device = torch.device('cuda')
    cluster = FaceCluster(cache_dir=cache_dir, device=device)
    faceidx = cluster.run(tracks)
    with open(os.path.join(cache_dir, 'pywork', 'faceidx.pkl'), 'wb') as f:
        pickle.dump(faceidx, f)

