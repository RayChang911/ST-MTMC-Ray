"""
Detector.py
    input: image
    output: bboxes, (confidence score), (mask) ......etc
"""
import cv2
import numpy as np
import h5py
from keras.engine.training import Model
from keras.saving.save import load_model
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image

from CONFIGS import *
from utils.utils import objectCrop

class Detection(object):
    def __init__(self, ltwh, crop, confidence=0, cls=None, feature=None):
        self.ltwh = np.asarray(ltwh, dtype=np.float)
        self.crop = crop
        self.confidence = confidence
        self.cls = cls
        self.feature = feature

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.ltwh.copy()
        ret[2:] += ret[:2]
        return ret
        
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.ltwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_tuple(self):
        return (self.ltwh, self.confidence, self.cls)

class MOGDetector():
    def __init__(self, camid, Model, history: int = 30 * STREAM_FPS ) -> None:
        self.mog = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold = 90)
        self.camid = camid
        self.classify = ResNetClassifier(Model)
        
    def detect(self, frame) -> list[Detection]:
        mask = self.mog.apply(frame)
        # 基礎形態學強化
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=2)
        _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        # 萃取邊界
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for t, c in enumerate(cnts):
            # 忽略面積太大/小的區域
            if cv2.contourArea(c) < STREAM_SIZE[0]*STREAM_SIZE[1]/4500 or cv2.contourArea(c) > STREAM_SIZE[0]*STREAM_SIZE[1]/2:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            # MOG 無法判斷 class, confidence, feature
            crop = objectCrop(frame, [x, y, x+w, y+h])
            # 分類
            cls = self.classify.Classifier(crop)
            # Detection
            det = Detection((x,y,w,h), crop, 0, cls, 'None')
            dets.append(det)
        return dets

class ResNetClassifier():
    def __init__(self, Model) -> None:
        self.cls_list = ["car", "motorcycle", "person", "undefined"]

        self.model_resnet_classifier = Model.createInference_classifer()
    def Classifier(self, crop) -> str:
        # 分類器 Classifier
        img = cv2.resize(crop, (244, 244))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        MOG_res_feature = self.model_resnet_classifier.predict(img)[0]
        MOG_res_feature = [float(round(x, 3)) for x in MOG_res_feature]

        # 利用 fully connected layer分類
        if MOG_res_feature[np.argmax(MOG_res_feature)] >= 0.7:
            cls_num = np.argmax(MOG_res_feature)
        else:
            cls_num = 3
        return self.cls_list[cls_num]