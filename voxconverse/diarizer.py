#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
from typing import List, Tuple, Dict

import numpy as np
from scipy import signal
from scipy.ndimage import binary_closing
from sklearn.cluster import AgglomerativeClustering

import torch
import torch.nn.functional as F

from .utils import find_runs, majority_filter_traditional, VIA_JSON_TEMPLATE

class Diarizer:
    def __init__(self, cache_dir: str, out_dir: str) -> None:
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.frame_rate = 25
        self.dist_thres = 0.3
        self.spk_thres = 0.6
        self.max_length = 120  #  in minutes

        os.makedirs(self.out_dir, exist_ok=True)
        self.data = VIA_JSON_TEMPLATE

    def addvad(self, vadscore: np.ndarray, vad:List[Tuple[float, float]]) -> np.ndarray:
        """
        Add VAD results to the vadscore

        Args:
            vadscore: np.ndarray, vadscore
            vad: List[Tuple[float, float]], vad results (start time, duration)

        Returns:
            np.ndarray, vadscore
        """
        for v in vad:
            vs = int(v[0] * self.frame_rate)
            ve = int((v[0] + v[1]) * self.frame_rate)
            vadscore[vs:ve] += 1
        return vadscore

    def run(self, 
            tracks: List[Dict], 
            asdres: List[np.ndarray], 
            vads: Dict[str, List[Tuple[float, float]]], 
            face_id: List[int], 
            spkfeats: torch.Tensor, 
            origfile: str = '/Users/jaesung/Desktop/sample.mp4'):
        """
        Run diarization

        Args:
            tracks: List[Dict], list of tracks
            asdres: List[np.ndarray], list of ASD results
            vads: Dict[str, List[Tuple[float, float]]], dict of VAD results
            face_id: List[int], list of face ids
            spkfeats: torch.Tensor, speaker features
            origfile: str, original video file

        Returns:
            List[Tuple[float, float, str]], list of diarization results
        """
        logging.info("Running diarization using intermediate results...")

        # TODO : Fix this with the length of actual videofile
        max_length = self.max_length * 60 * self.frame_rate
        allvad = np.zeros(max_length)
        allvad = self.addvad(allvad, vads['audio.wav'])
        allvad_copy = np.copy(allvad)

        spkemb = {}
        spkseg = {}
        utterances = []

        for tidx, track in enumerate(tracks):
            mean_dists = np.mean(np.stack(asdres[tidx], 1), 1)
            minidx = np.argmin(mean_dists, 0)

            fdist = np.stack([dist[minidx] for dist in asdres[tidx]])
            fdist = np.pad(fdist, (3, 3), 'constant', constant_values=10)

            fconf = np.median(mean_dists) - fdist
            fconfm = signal.medfilt(fconf, kernel_size=25)

            vadscore = np.zeros_like(fconfm)
            vadscore = self.addvad(vadscore, vads[f'{tidx:05d}.wav'])

            vadoutput = np.copy(vadscore)

            if face_id[tidx] not in spkseg:
                spkseg[face_id[tidx]] = np.zeros(max_length)

            spkseg[face_id[tidx]][track['track']['frame']] += 1

            vadoutput = np.copy(vadscore)

            # This is heuristic
            vadoutput[fconfm > 0.5] += 1
            vadoutput[fconfm > 0.9] += 1

            vadoutput[vadoutput < 2] = 0
            vadoutput[vadoutput >= 2] = 1

            vadoutput = binary_closing(vadoutput, structure=np.ones((12))).astype(int)

            allvad_copy[track['track']['frame'][0]:track['track']['frame'][0] + len(vadscore)] -= vadoutput

            run_v, run_s, run_l = find_runs(vadoutput)

            run_s += track['track']['frame'][0]

            for r_idx, r_v in enumerate(run_v):
                if r_v > 0 and run_l[r_idx] > 15:
                    time_s = float(run_s[r_idx]) / self.frame_rate
                    time_e = float(run_s[r_idx] + run_l[r_idx]) / self.frame_rate
                    utterances.append({'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 's': face_id[tidx]})

                    midtime = (time_s + time_e) / 2
                    midfeat = min(max(0, (midtime * 5) - 5), len(spkfeats) - 1)

                    if face_id[tidx] not in spkemb:
                        spkemb[face_id[tidx]] = []
                    spkemb[face_id[tidx]].append(spkfeats[[midfeat]])

        keys = list(spkemb.keys())
        keys.sort()

        for k in keys:
            spkemb[k] = torch.mean(torch.cat(spkemb[k], 0), 0, keepdim=True)
        embmat = torch.cat([spkemb[k] for k in keys], 0)
        simmat = F.cosine_similarity(
            embmat.unsqueeze(-1).expand(-1, -1, len(embmat)),
            embmat.unsqueeze(-1).expand(-1, -1, len(embmat)).transpose(0, 2)
        ).detach().cpu().numpy()

        ovlmat = []
        for k in keys:
            ovlmat.append([np.sum(spkseg[k] * spkseg[q]) for q in keys])
        ovlmat = np.array(ovlmat).astype(int).clip(0, 1)
        ovlmat[range(0, len(ovlmat)), range(0, len(ovlmat))] = 0

        if len(ovlmat) >= 2:
            ovl_penalty = 100
            agc = AgglomerativeClustering(
                n_clusters=None, metric='precomputed', distance_threshold=self.dist_thres, linkage='average'
            ).fit(1 - simmat + ovlmat * ovl_penalty)
            labels = agc.labels_
        else:
            labels = np.array([0])

        labelname = {}
        groupname = {}
        for gidx, label in enumerate(range(max(labels) + 1)):
            idxlist = [keys[x] for x in np.where(labels == label)[0].tolist()]
            groupname[gidx] = 'ID_' + '/'.join(map(str, idxlist))
            for iidx in idxlist:
                labelname[iidx] = 'ID_' + '/'.join(map(str, idxlist))

        for uttidx, utterance in enumerate(utterances):
            utterance['av'] = {'1': labelname[utterance['s']]}
            del utterance['s']
            self.data['metadata'][f'{uttidx}'] = utterance

        allvad_copy[allvad_copy < 1] = 0
        run_v, run_s, run_l = find_runs(allvad_copy)
        for r_idx, r_v in enumerate(run_v):
            if r_v > 0 and run_l[r_idx] > 10:
                indices = []
                for frame in range(run_s[r_idx], run_s[r_idx] + run_l[r_idx]):
                    fr_feat = int(min(max(0, (frame / 5) - 5), len(spkfeats) - 1))

                    cossim = F.cosine_similarity(embmat, spkfeats[[fr_feat]])

                    mval = torch.max(cossim)
                    midx = torch.argmax(cossim)

                    if mval >= self.spk_thres:
                        indices.append(labels[midx])
                    else:
                        indices.append(-1)

                indices = majority_filter_traditional(indices, 25)

                run_vs, run_ss, run_ls = find_runs(indices)

                for rs_idx, rs_v in enumerate(run_vs):
                    time_s = float(run_s[r_idx] + run_ss[rs_idx]) / self.frame_rate
                    time_e = float(run_s[r_idx] + run_ss[rs_idx] + run_ls[rs_idx]) / self.frame_rate

                    if rs_v != -1:
                        self.data['metadata'][f'{r_idx}_{rs_idx}'] = {
                            'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 'av': {'1': groupname[rs_v]}
                        }
                    else:
                        self.data['metadata'][f'{r_idx}_{rs_idx}'] = {
                            'vid': '1', 'flg': 0, 'xy': [], 'z': [time_s, time_e], 'av': {'1': 'unknown'}
                        }

        self.data["file"]["1"]["src"] = str(origfile)

        # Store the json file
        jsonfile = os.path.join(self.out_dir, 'result.json')
        with open(jsonfile, 'w') as outfile:
            json.dump(self.data, outfile)

        # Store the rttm file
        result = []
        bname = os.path.basename(origfile).split('.')[0]
        rttmfile = os.path.join(self.out_dir, 'result.rttm')
        with open(rttmfile, 'w') as f:
            uttkeys = list(self.data['metadata'].keys())
            uttkeys.sort()

            for uttkey in uttkeys:
                utt = self.data['metadata'][uttkey]
                if len(utt['z']) == 2:
                    spk = utt['av']['1'].replace(' ', '_')
                    start = utt['z'][0]
                    end = utt['z'][1]
                    f.write(f'SPEAKER {bname} 1 {start:.2f} {end - start:.2f} <NA> <NA> {spk} <NA> <NA>\n')
                    result.append((start, end, spk))

        return result


if __name__ == '__main__':
    cache_dir = '/users/jaesung/voxconverse_method/temp'
    out_dir = '/users/jaesung/voxconverse_method/temp'
    with open('/users/jaesung/voxconverse_method/temp/pywork/tracks.pckl', 'rb') as f:
        tracks = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/activesd.pckl', 'rb') as f:
        asdres = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/webrtc.pkl', 'rb') as f:
        vads = pickle.load(f)
    with open('/users/jaesung/voxconverse_method/temp/pywork/faceidx.pkl', 'rb') as f:
        faceclusters_idx = pickle.load(f)
    spkfeats = torch.load('/users/jaesung/voxconverse_method/temp/pywork/ecapa.pt')
    diarizer = Diarizer(cache_dir=cache_dir, out_dir=out_dir)
    result = diarizer.run(tracks, asdres, vads, faceclusters_idx, spkfeats)
