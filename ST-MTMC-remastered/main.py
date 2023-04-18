"""
    Basic usage of all classes
"""
import cv2
import numpy as np
import time
import os
import json
import logging
from threading import Thread
from queue import Queue
from copy import deepcopy

import utils.utils as utils
from src.streaming import StreamHandler
from src.detector import MOGDetector
from src.tracker import DeepSORTTracker, Target
from src.reid import REIDHandler
from utils.mAHT import Transformer
from utils.reidModel import ReidModel
# from utils.reidModel import featureExtractor
from CONFIGS import *

logging.basicConfig(level=logging.DEBUG)

class FramePackage():
    """Contains info that factory need from one frame.
    """
    def __init__(self, camid: str, frame, timestamp:float, targets: dict[str, Target], activeTids: list[str]) -> None:
        self.camid = camid
        self.timestamp = timestamp
        self.frame = frame
        self.targets = targets
    
    def asDict(self):
        return {
                'camid': self.camid,
                'timestamp': self.timestamp,
                'frame': self.frame,
                'targets': self.targets
            }

class Worker():
    """
    Streaming -> Detecting -> Tracking
    
    output: Frame Data
    """
    def __init__(self, camid: str, show_flag: bool=True, save_flag: bool= True) -> None:
        self.dismiss = False
        self.camid = camid
        self.show_flag = show_flag
        self.save_flag = save_flag
        self.transformer = Transformer(camid)
        self.stream = StreamHandler(camid, async_flag=False, save_flag=save_flag)
        self.stream.start()
        self.ReidModel = ReidModel()
        self.detector = MOGDetector(camid, self.ReidModel)
        self.targetDict: dict[str, Target] = {}
        self.tracker = DeepSORTTracker(camid, self.targetDict)
        # self.featureExtractor = featureExtractor()
        self.output = Queue(1)

    def processOneFrame(self) -> FramePackage:
        """Main working function"""
        streamData = self.stream.getLatest()
        self.frame = streamData['frame']
        self.timestamp = streamData['count']
        canvas = self.frame.copy()
        dets = self.detector.detect(self.frame)
        activeTids, crops = self.tracker.track(dets, self.timestamp, self.frame, self.ReidModel)

        if self.show_flag or self.save_flag:
            utils.drawDetections(canvas, dets)
            utils.drawActiveTarget(canvas, self.tracker.targetDict, activeTids)
        if self.show_flag:
            # cv2.imshow(f'Streaming {self.camid}', self.frame)
            cv2.imshow(f'Canvas of {self.camid}', canvas)
            cv2.waitKey(1)
        if self.save_flag:
            for i, crop in enumerate(crops):
                self.saveTargetCrop(activeTids[i], crop)
            self.stream.writeFrame(canvas)

        return FramePackage(self.camid, self.timestamp, self.frame, self.tracker.targetDict, activeTids)

    def saveTargetCrop(self, tid: str, crop: np.ndarray):
        # check if folder existed
        if not os.path.exists(os.path.join(TRACKLETS_PATH, tid)):
            os.mkdir(os.path.join(TRACKLETS_PATH, tid))
        cv2.imwrite(os.path.join(TRACKLETS_PATH, tid, f'{self.timestamp}.jpg') , crop)

    def work(self):
        """Threading function"""
        while not self.dismiss and not self.stream.stopped:
            # if self.output.full():
            #     self.output.get()
            package = self.processOneFrame()
            """
            Do stuff to package before hand it to factory
            """
            self.output.put(package)

        # End of the work
        logging.info(f'Worker of cam {self.camid} dismissed.')
        if self.save_flag:
            self.stream.save.release()


class Factory():
    """A factory collect frame package from worker and assemble product"""
    def __init__(self, workers: dict[str, Worker], save_flag: bool = True) -> None:
        self.workers = workers
        self.stopped = False
        self.save_flag = save_flag
        self.floorplan = cv2.imread(FLOOR_PATH)
        self.canvas = deepcopy(self.floorplan)
        if save_flag:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.save = cv2.VideoWriter(
                os.path.join(OUT_PATH, f'monitor.mp4'), 
                fourcc, 
                30,
                self.floorplan.shape[0:2][::-1]
            )
    
    def collectAll(self) -> dict[str, FramePackage]:
        """Collect frame package from all worker"""
        packages = {}
        # collect framepackages from workers
        for workerid, worker in self.workers.items():
            packages[workerid] = worker.output.get()
        return packages

    def saveTargetsLog(self, Targets: list[Target]):
        with open(OBJLOG_FILE, 'r') as fp:
            obj_log = json.load(fp)
        for t in Targets:
            obj_log[t.tid] = {
                'starttime': t.starttime,
                'timestamps': t.timestamps,
                'framecount': t.count,
                'location': t.locations
            }
            del self.workers[t.camid].targetDict[t.tid]
        with open(OBJLOG_FILE, 'w') as fp:
            json.dump(obj_log, fp)

    def determineCollisionCondition(self, Groups, collisionth = 30., predictTime = 60):
        Collisionlist = []
        for gid1, group1 in Groups:
            print(group1)
            predict_pos1 = []
            for i in range(2):
                new_value = group1.location[-1][i] + predictTime * group1.vector[-1][i]
                predict_pos1.append(new_value)

            for gid2, group2 in Groups:
                if gid1 >= gid2 or group1.active == False or group2.active == False or (
                        group1.cls == "person" and group2.cls == "person"):
                    continue
                else:
                    predict_pos2 = []
                    for i in range(2):
                        new_value = group2.location[-1][i] + predictTime * group2.vector[-1][i]
                        predict_pos2.append(new_value)

                    distance_of_obj_pre = np.linalg.norm(np.array(predict_pos1) - np.array(predict_pos2))
                    if distance_of_obj_pre <= collisionth:
                        print("collision condition -------------------------------")
                        print(gid1)
                        print(gid2)
                        print("distance = " + str(distance_of_obj_pre))
                        Collisionlist.append([group1, group2])
        if Collisionlist:
            return True
        else:
            return False


    def start(self) -> None:
        """Factory
            Collect and do stuff with package from all worker, calculate direction/speed for targers example
            Prevent Race Condition, I/O common file here
        """
        self.REID = REIDHandler()
        while not self.stopped:
            """
            Do stuff with package from all worker, calculate direction/speed for targers example
            Prevent Race Condition, I/O common file here
            """
            canvas = deepcopy(self.floorplan)
            packages = self.collectAll()
            
            inactiveTargets = []
            ungrouppedTargets = []
            for camid, package in packages.items():
                ## Extract targets' id that haven't been groupped
                for tid, target in list(package.targets.items()):
                    if target.active == False:
                        inactiveTargets.append(target)

                    print(target.tid)
                    print(target.cls)
                    print(target.count)
                    print(target.gid if target.gid else "None")
                    print("========================================")
                    # print(target.vectors)
                    # print(target.location)
                    # print(target.feet)
                    if target.gid:                      # 已分類
                        self.REID.groups[target.gid].update(target)

                    ## Ignore target exist not long enough
                    if not target.gid and target.count >= 30:  # 未分類，存在大於<30>個frame開始算
                        ungrouppedTargets.append(target)
            if ungrouppedTargets:
                self.REID.reidentify(ungrouppedTargets)
            self.saveTargetsLog(inactiveTargets)

            # determine whether collsion
            collisionBool = self.determineCollisionCondition(self.REID.groups.items())
            logging.info(f'collision condition: {str(collisionBool)}')

            """
            Draw and Show
            """
            for camid, package in packages.items():
                utils.drawPackage(canvas, package)
            cv2.imshow(f'Monitor of {list(self.workers.keys())}', canvas)
            if cv2.waitKey(1) == 27:
                break
            if self.save_flag:
                self.save.write(canvas)
        self.close()

    def close(self):
        self.dimissAllWorker()
        time.sleep(3)
        if self.save_flag:
            self.save.release()
        logging.debug('Factory End')

    def dimissAllWorker(self):
        for worker in self.workers.values():
            worker.dismiss = True
            if worker.output.full():
                worker.output.get()


if __name__ == '__main__':
    threads = []
    workers = {}

    """
    Initial Check
    """
    with open(OBJLOG_FILE, 'w') as fp:
        json.dump({}, fp)
    
    """
    Initiate Workers
    """
    for cam in CAMERAS_LIST:
        worker = Worker(cam, show_flag=True, save_flag=True)
        workers[cam] = worker
        threads.append(Thread(target = worker.work, args = ()))
        # threads.append(Process(target = worker.work, args = (True,)))
    for t in threads:
        t.start()

    factory = Factory(workers)
    factory.start()
    cv2.destroyAllWindows()

    # Recycle all threads
    for t in threads:
        t.join()