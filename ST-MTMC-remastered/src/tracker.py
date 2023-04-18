"""
Tracker.py
    input: bbox
    output: bbox with id
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import numpy as np
import logging
import os
import cv2

from src.detector import Detection
from utils.mAHT import Transformer
from utils.utils import objectCrop
from utils.reidModel import inference
from utils.utils import calcObjAspect

from CONFIGS import *

def BBox2Foot(ltrb) -> tuple[float, float]:
    return (ltrb[0]+0.5*(ltrb[2]-ltrb[0]), ltrb[3])

def calc_vectors(buffer: list) -> list:
    if len(buffer) > 0:
        return np.average(np.asarray(buffer), axis=0).tolist()
    else:
        return [0., 0.]

class Target():
    def __init__(self, tid: str, cls, camid: str, starttime: "int|float", ltrb, tensor, transformer: Transformer) -> None:
        self.init_flag = True
        self.active = True
        self.tid = tid
        self.cls = cls
        self.camid = camid
        self.transformer = transformer
        self.cameraVector = self.transformer.calculateCameraVec()
        self.starttime = starttime
        self.timestamps = []
        self.bboxes = []
        self.feets = []
        self.locations = []
        self.vectors = []
        self.vectors_buffer = []
        self.directions = []
        self.tensors = []
        self.count = 1
        self.update(ltrb, starttime, tensor)
        self.gid = ''

    def __str__(self) -> str:
        return f'Target {self.tid}'
    
    def update(self, ltrb, timestamp: "int|float", tensor) -> None:
        self.bboxes.append(ltrb)
        self.timestamps.append(timestamp)
        foot = BBox2Foot(ltrb)
        self.feets.append(foot)
        self.locations.append(list(map(float, self.transformer.transform(foot, inv = False))))

        if not self.init_flag:
            if len(self.vectors_buffer) >= 10:
                self.vectors_buffer = self.vectors_buffer[1:]
            self.vectors_buffer.append(np.asarray([
                self.locations[-1][0] - self.locations[-2][0],
                self.locations[-1][1] - self.locations[-2][1]]))
        else: 
            self.init_flag = False

        vector = calc_vectors(self.vectors_buffer)
        self.vectors.append(vector)

        direction_deg = calcObjAspect(vector, self.cameraVector)
        self.directions.append(direction_deg)

        self.count += 1

        self.tensors.append(tensor)
        '''
        if len(self.tensors) > 20: 
            self.tensors = self.tensors[1:]
        '''

class DeepSORTTracker():
    def __init__(self, camid: str, targetDict: dict[str, Target]) -> None:
        self.core = DeepSort(max_age=5)
        self.targetDict: dict[str, Target] = targetDict
        self.camid = camid
        self.transformer = Transformer(camid)
        # self.ftExtractor = featureExtractor()

    def track(self, dets: list[Detection], timestamp: "int|float", frame, ReidModel) -> tuple[list[str], list[np.ndarray]]:
        """
            return active target id list
        """
        bboxes = []
        for det in dets:
            bboxes.append(det.to_tuple())
        if bboxes == []:
            self.checkEnd([], timestamp)
            return [], []
        
        # actives: [tid, crop]
        activeTids: list[str] = []
        crops: list[np.ndarray] = []
        tracks = self.core.update_tracks(bboxes, frame=frame)
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            track_id = f'{self.camid}_{track.track_id}'
            ltrb = track.to_ltrb()
            crop = objectCrop(frame, ltrb)
            if 0 in crop.shape:
                continue
            tensor = inference(crop, track.det_class, ReidModel)
            if track_id in self.targetDict:
                self.targetDict[track_id].update(ltrb, timestamp, tensor)
            else:
                self.targetDict[track_id] = Target(
                    track_id,
                    track.det_class,
                    self.camid, 
                    timestamp, 
                    ltrb,
                    tensor,
                    self.transformer
                )
            activeTids.append(track_id)
            crops.append(crop)

        self.checkEnd(activeTids, timestamp)
        return activeTids, crops
    
    def checkEnd(self, activeTids: list[str], timestamp:"int|float"):
        for tid, target in self.targetDict.items():
            if tid not in activeTids and target.timestamps[-1]:
                target.active = False