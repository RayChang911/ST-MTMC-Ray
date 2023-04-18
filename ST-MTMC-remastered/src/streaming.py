"""
streaming.py
    All stuff regarding the video/live streaming


    single frame template:
    {
        'timestamp':
        'frame':
    }

"""
from threading import Thread
from queue import Queue
import cv2
import time
import os

from CONFIGS import *
from src.detector import MOGDetector

class StreamHandler(object):
    def __init__(self, camid: str, async_flag: bool=False, save_flag: bool=True) -> None:
        self.camid = camid
        self.async_flag = async_flag
        self.stopped = False
        self.queue = Queue(1)
        self.count = 0
        self.cap = cv2.VideoCapture(f'{VIDEOS_PATH}/{self.camid}.mp4')
        self.ret, self.frame = self.cap.read()
        self.thread = Thread(target=self.streaming)
        if save_flag:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.save = cv2.VideoWriter(
                os.path.join(OUT_PATH, f'{self.camid}.mp4'), 
                fourcc, 
                self.cap.get(cv2.CAP_PROP_FPS), 
                (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

    def start(self) -> None:
        self.thread.start()

    # Recieve and Discard
    def streaming(self) -> None:
        while not self.stopped and self.ret:
            if self.queue.full() and self.async_flag:
                # If the queue is full, discard the old one (Process time > frame rate)
                self.queue.get()

            self.ret, self.frame = self.cap.read()
            if not self.ret:
                self.stopped = True
                break
            streamData = {
                'timestamp': time.time(),
                'count': self.count,
                'frame': self.frame
            }
            self.queue.put(streamData)
            self.count += 1
            time.sleep(1/STREAM_FPS)

    def writeFrame(self, canvas) -> None:
        self.save.write(canvas)

    def getLatest(self) -> dict:
        return self.queue.get()