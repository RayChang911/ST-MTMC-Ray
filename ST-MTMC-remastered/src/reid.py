"""
reid.py
    Assigning group to target.
    All Target should have group at the end.
"""
import json
import os
import numpy as np
import logging
import torch.nn as nn
import utils.utils
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

from src.tracker import Target
from utils.routeNet import RouteNetwork
from utils.utils import distance, unitVector, calcAcuteAngle
from utils.reidModel import ReidModel, inference, compareTwo
from CONFIGS import *

class Group():
    """
    Group is considered the targets with same identity in multiple cameras along time
    a group shouldn't appear in different locations at same time
    """
    def __init__(self, gid: str, firstTid: str, cls: str, timestamps: [], location: [], vector:[], direction:[], tensor:[]) -> None:
        self.gid = gid
        self.tids: list[str] = [firstTid]
        self.cls = cls
        self.timestamps = timestamps
        self.location = location
        self.vector = vector
        self.direction = direction
        self.tensor = tensor

        ######
        self.active = True

    def __str__(self) -> str:
        return f'{self.gid}:{self.cls}:{self.tids}:{self.timestamps}:{self.location[-1]}:{self.vector[-1]}:{self.direction[-1]}, {self.active}'
        # return f'{self.gid}:{self.cls}:{self.tids}:{self.timestamps}:{len(self.tensor)}:{len(self.vector)}:{len(self.direction)}, {self.active}'
    def addNewTarget(self, target: Target):
        self.tids.append(target.tid)
        target.gid = self.gid

    def stop(self):
        self.active= False

    def update(self, target):
        ## update from active target in package
        self.timestamps = target.timestamps
        self.location = target.locations
        self.vector = target.vectors
        self.direction = target.directions
        self.tensor = target.tensors


    def selfCorrection(self):
        activeTargets = [t for t in self.targets if t.active == True]
        print(f'Group: {self.gid}')
        for target in activeTargets:
            print(f'active: {target.tid}, {target.location[-1]}')

class REIDHandler():
    def __init__(self, score_th: float=0.77) -> None:
        self.newid = 0
        self.score_th = score_th
        self.routeNet = RouteNetwork(os.path.join(ROOT_PATH, 'routeNet.json'))
        self.groups: dict[str, Group] = {}


    def reidentify(self, ungrouppedTargets: list[Target], th= 1.2, glostth = 300) -> None: # STOP = 5S
        """
        Main REID function:
            compare target to existed active "groups" (not target) ?
        """

        activeGroupIds = [gid for gid, group in self.groups.items() if group.active == True]
        # print("ungrouppedTargets")
        # print(ungrouppedTargets)
        # print("activeGroupIds")
        # print(activeGroupIds)
        scoreMatrix = []
        for target in ungrouppedTargets:
            ## First target, init first group for it
            if activeGroupIds == []:
                self.createNewGroup(target)
                activeGroupIds.append(target.gid)
                break
            score = []
            print(target.tid + "---------------------------------------------------------------")
            print(target.cls)
            for gid in activeGroupIds:
                ## Applying all reid methods
                print("Groups **************************")
                print(self.groups[gid])

                # group太久沒更新: active = False
                if target.timestamps[-1] - self.groups[gid].timestamps[-1] >= glostth:
                    print("group lost")
                    self.groups[gid].stop()
                    score.append(0.)
                    continue

                # 不同class的話不做reid
                if target.cls != self.groups[gid].cls:
                    print("Not the same class")
                    score.append(0.)
                    continue

                # Geo_match Test
                goematchScore = self.geoMatching(target, self.groups[gid])
                print("goematchScore")
                print(goematchScore)
                # STP Test
                stpScore = self.stpMatching(target, self.groups[gid])
                print("stpScore")
                print(stpScore)
                # RBG Test
                rgbScore = self.rgbMatching(target, self.groups[gid])
                print("rbgScore")
                print(rgbScore)

                score.append(goematchScore + stpScore + rgbScore)
            scoreMatrix.append(score)

        if not scoreMatrix:
            return
        print("scoreMatrix")
        print(scoreMatrix)

        results = self.solveScoreMatrix(scoreMatrix)

        for tIndex, gIndex in results:
            if scoreMatrix[tIndex][gIndex] > th:
                logging.info(f'scores: {scoreMatrix[tIndex]}')
                logging.info(f'matched: {activeGroupIds[gIndex]} & {ungrouppedTargets[tIndex].tid}')
                self.groups[activeGroupIds[gIndex]].addNewTarget(ungrouppedTargets[tIndex])
            else:
                self.createNewGroup(ungrouppedTargets[tIndex])

    def createNewGroup(self, target:Target):
        self.groups[str(self.newid)] = Group(
            str(self.newid),
            target.tid,
            target.cls,
            target.timestamps,
            target.locations,
            target.vectors,
            target.directions,
            target.tensors
        )
        target.gid = str(self.newid)
        self.newid += 1

    def saveGroups(self):
        with open(GROUPS_FILE, 'r+') as fp:
            try:
                data = json.load(fp)
            except json.decoder.JSONDecodeError:
                data = {}
        for gid, group in self.groups.items():
            if not group.active:
                data[gid] = [target.tid for target in group.targets]
        with open(GROUPS_FILE, 'w') as fp:
            json.dump(data, fp, indent=4)


    def geoMatching(self, target: Target, group: Group, th = 50) -> float: # 達到th = 相同物件 = goe-matching滿分 th要改!!!!!!!!!!!!!!!!!!!!!!
        """
        use location to determine whether the same group, merge into STP
        """
        if target.timestamps[-1] != group.timestamps[-1]:
            return 0.
        else:
            geo_d = distance(target.locations[-1], group.location[-1])
            geoMatchingPoint = th / geo_d
            if geoMatchingPoint <= 1.:
                return geoMatchingPoint
            else:
                return 1.

    def stpMatching(self, target: Target, group: Group) -> float:
        """
        STP require delta_t, distance, velocity
        """
        target2 = group
        delta_t = target.timestamps[-1] - target2.timestamps[-1]
        
        paths, d = self.routeNet.NDistance(target.locations[-1], target2.location[-1])
        if len(paths) <= 2:
            d = distance(target.locations[-1], target2.location[-1])
        v = float(np.linalg.norm(target.vectors[-1]))
        if delta_t > 300:
            return 0.
        elif delta_t <= 1:
            if d > 30 or target.tid[0] == target2.tids[-1][0]:
                return 0.
            else:
                return 1.
        else:
            return STP(delta_t, d, v)

    def rgbMatching(self, target: Target, group: Group, degth = 90):
        featureSims = []
        for i, direction1 in enumerate(target.directions) :
            print("target.tid")
            print(target.tid)
            print(target.timestamps[i])
            print(direction1)
            for j, direction2 in enumerate(group.direction):
                delta_deg = calcAcuteAngle(direction1, direction2)
                if delta_deg <= degth:
                    try:
                        featureSims.append(compareTwo(target.tensors[i], group.tensor[j]))
                        print("group.tid")
                        print(group.tids)
                        print(group.timestamps[j])
                        print(direction2)
                    except IndexError:
                        continue

        try:
            avg = sum(featureSims) / len(featureSims)
        except ZeroDivisionError:
            avg = 0.
        return avg

    def solveScoreMatrix(self, scoreMatrix: list[list[float]]) -> list[tuple[int, int]]:
        """
        Applying Hungarian Algorithm
        return [(tindex, gindex), ...]
        """
        scoreMatrix = np.asarray(scoreMatrix)
        tIndex, gIndex = linear_sum_assignment(scoreMatrix, maximize = True)
        return list(zip(tIndex, gIndex))

def STP(delta_t: "int|float", d: "int|float", v: "int|float") -> float:

    v = float(v)
    mu = delta_t * v
    """
    sigma = delta * tolerant time * fps
    """
    sigma = min(delta_t, 60) * v # 2s
    # sigma = min(delta_t, 300) * v # 10s

    if v == 0.:
        return 0.
    if d > mu:
        return (np.exp(- (d - mu)**2 / (sigma**2)))
    else: # d <= mu
        return 1.

        