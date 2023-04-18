import cv2
import numpy as np
import logging
from keras.engine.training import Model
from keras.saving.save import load_model


logging.basicConfig(level=logging.INFO)

from CONFIGS import *

def distance(pt1: list, pt2: list) -> float:
    d = np.linalg.norm(np.array(pt1)-np.array(pt2))
    return float(d)

def unitVector(vector: list):
    if vector[0] == 0 and vector[1] == 0:
        return np.array([0., 0.])
    else:
        return vector / np.linalg.norm(vector)

def calcObjAspect(obj_vec, cam_vec):
    v1 = unitVector(cam_vec)
    v2 = unitVector(obj_vec)
    # rad = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    # deg = np.rad2deg(rad)
    ang1 = np.arctan2(*v1[::-1])
    ang2 = np.arctan2(*v2[::-1])
    deg = np.rad2deg((ang2 - ang1) % (2 * np.pi))
    return deg

def calcAcuteAngle(deg1, deg2) -> float:
    dif = abs(deg1 - deg2)
    if dif > 180.:
        return float(360. - dif)
    else:
        return float(dif)


def objectCrop(img: np.ndarray, ltrb: list) -> np.ndarray:
    if ltrb[0] < 0:
        ltrb[0] = 0
    if ltrb[1] < 0:
        ltrb[1] = 0
    if ltrb[2] > STREAM_SIZE[0]:
        ltrb[2] = STREAM_SIZE[0]
    if ltrb[3] > STREAM_SIZE[1]:
        ltrb[3] = STREAM_SIZE[1]
    obj_img = img[int(ltrb[1]):int(ltrb[3]), int(ltrb[0]):int(ltrb[2])]
    return obj_img

def drawPoint(canvas, point: list, color = (0, 0, 255), text: str='', withCircle = False):
    cv2.circle(
        canvas, 
        tuple([int(point[0]), int(point[1])]), 
        5, 
        color, 
        -1
    )
    if withCircle:
        cv2.circle(
        canvas, 
        tuple([int(point[0]), int(point[1])]), 
        20, 
        color, 
        1
    )
    if text:
        cv2.putText(
                canvas,
                str(text),
                tuple([int(point[0]), int(point[1])]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color
            )

def drawDetections(canvas, dets: list):
    for det in dets:
        ltrb = det.to_tlbr()
        canvas = cv2.rectangle(
            canvas, (int(ltrb[0]), 
            int(ltrb[1])), 
            (int(ltrb[2]), 
            int(ltrb[3])), 
            (255, 0, 0),
            2
        )

def drawActiveTarget(canvas, targetDict: dict, activeTids: list[str]):
    for tid in activeTids:
        try:
            target = targetDict[tid]
        except KeyError:
            continue
        canvas = cv2.putText(
            canvas, 
            f'{target.cls}-{tid}',
            (int(target.bboxes[-1][0]), int(target.bboxes[-1][1])), 
            0,
            1e-3 * canvas.shape[0], 
            (0, 255, 0),
            1
        )
        canvas = cv2.rectangle(
            canvas, 
            (int(target.bboxes[-1][0]), 
            int(target.bboxes[-1][1])), 
            (int(target.bboxes[-1][2]), 
            int(target.bboxes[-1][3])), 
            (0, 255, 0),
            2
        )

def drawPackage(canvas, package):
    for tid, target in list(package.targets.items()):
        point = target.locations[-1]
        vec = target.vectors[-1]
        arrow_pt = tuple(map(int, (point[0] + 10*vec[0], point[1] + 10*vec[1])))
        point = list(map(int, point))
        color = (0, 0, 255) if target.active == True else (0, 0, 0)
        drawPoint(canvas, point, color, f'{target.gid if target.gid else "None"}-{target.cls + tid}', withCircle=True)
        cv2.line(canvas, (point[0], point[1]), arrow_pt, (0, 0, 255), 2)

def loadReidModel(path: str):
    model = load_model(path,compile=False)  # Resnet Model
    model = Model(inputs=model.input, outputs=model.layers[
        len(model.layers) - 3].output) # 共 178 layer，去除後面兩層
    logging.info('Load ResNet50 Reid Model')
    return model