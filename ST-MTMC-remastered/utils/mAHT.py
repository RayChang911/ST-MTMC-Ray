import sys
sys.path.append('./src/')
import numpy as np
import cv2
import json
import os
import copy
from CONFIGS import *

exception_template = {
    'maskPath': '',
    'matrix':[]
}

def getPosition(event, x, y, flags, Coors: list):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('The image coordinate is {}, {}'.format(x, y))
        Coors.append([x, y])

def calcMatrix(cameraCoors: list, floorCoors: list) -> list:
    hm, status =  cv2.findHomography(np.array(cameraCoors, dtype='float32'), np.array(floorCoors, dtype='float32'))
    # invhm = np.linalg.inv(hm)
    return hm

def drawPoint(i, point: list, img, c_size: int=10, t_size: int=1):
    point = list(map(int, point))
    CENTER = tuple(point)
    cv2.circle(img, CENTER, c_size, (127,0,127), -1)
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = t_size
    TEXT_THICKNESS = 1
    TEXT = str(i)
    text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (int(CENTER[0] - text_size[0] / 2), int(CENTER[1] + text_size[1] / 2))
    cv2.putText(img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)

def pickCoorPairs(n: int = 4, camImgPath: str = '', floorImgPath: str = '') -> tuple[list, list]:
    cameraCoors = []
    floorCoors = []
    camImg = cv2.imread(camImgPath)
    cv2.namedWindow('camImg')
    cv2.setMouseCallback('camImg', getPosition, cameraCoors)
    while True:
        cv2.imshow('camImg', camImg)
        if cv2.waitKey(0) == 27:
            break
    print('Camera Points: ', cameraCoors)
    for i, p in enumerate(cameraCoors):
        drawPoint(i, p, camImg)
    cv2.namedWindow('floorImg')
    cv2.setMouseCallback('floorImg', getPosition, floorCoors)
    while True:
        cv2.imshow('camImg', camImg)
        cv2.imshow('floorImg', cv2.imread(floorImgPath))
        if cv2.waitKey(0) == 27:
            break
    print('Floor Points', floorCoors)

    return cameraCoors, floorCoors

class Transformer(object):
    def __init__(self, camid: str, new = False) -> None:
        self.camid = camid
        self.path = os.path.join(CAMERAS_PATH, camid)
        self.masks = {}
        if new:
            self.default = []
            self.exceptions = {}
            self.new()
        else:
            self.default, self.exceptions = self.loadFile()
            self.default = np.array(self.default, dtype=np.float32)
            for k, v in self.exceptions.items():
                v['matrix'] = np.array(v['matrix'], dtype=np.float32)
                self.masks[k] = cv2.cvtColor(cv2.imread(v['maskPath']), cv2.COLOR_BGR2GRAY)

    def __str__(self) -> str:
        tmp_str = f'{self.default}\n{self.exceptions}\n{self.masks}'
        return tmp_str

    def new(self):
        self.setDefault()
        exceptionList = [x for x in os.listdir(self.path) if 'E-' in x]
        for E in exceptionList:
            self.setException(str(E.split('-')[1].split('.jpg')[0]))

    def loadFile(self) -> tuple[list, dict]:
        with open(self.path + '/transformer.json', 'r') as fp:
            tmp = json.load(fp)
        # print(tmp)
        return tmp['default'], tmp['exceptions']

    def setDefault(self) -> None:
        print(f'Setting default Matrix for cam: {self.camid}')
        cameraCoors, floorCoors = pickCoorPairs(4, self.path + '/empty.jpg', FLOOR_PATH)
        self.default = calcMatrix(cameraCoors, floorCoors)
        cv2.destroyAllWindows()

    def setException(self, exceptionName: str) -> None:
        print(f'Setting exception: {exceptionName} Matrix for cam: {self.camid}')

        cameraCoors, floorCoors = pickCoorPairs(4, self.path + '/empty.jpg', FLOOR_PATH)
        newException = exception_template.copy()
        newException['maskPath'] = self.path + f'/E-{exceptionName}.jpg'
        newException['matrix'] = calcMatrix(cameraCoors, floorCoors)

        mask = cv2.imread(newException['maskPath'])
        self.masks[exceptionName] = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        self.exceptions[exceptionName] = newException
        cv2.destroyAllWindows()

    def saveFile(self) -> None:
        saveDict = {
            'default': self.default.copy().tolist(),
            'exceptions': copy.deepcopy(self.exceptions)
        }
        for eName, eDict in saveDict['exceptions'].items():
            saveDict['exceptions'][eName]['matrix'] = saveDict['exceptions'][eName]['matrix'].tolist()
        with open(self.path + '/transformer.json', 'w+') as fp:
            json.dump(saveDict, fp, indent=4)
    def calculateCameraVec(self):
        middle_pt = [STREAM_SIZE[0] // 2, STREAM_SIZE[1] // 2]
        middle_pt2 = [middle_pt[0], middle_pt[1] - 300]
        pt1 = self.transform(middle_pt)
        pt2 = self.transform(middle_pt2)
        vec = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
        return vec

    def transform(self, point: tuple["int|float", "int|float"], inv = False) -> list[float]:
        M = self.default
        for eName, eDict in self.exceptions.items():
            try:
                if self.masks[eName][int(point[1])][int(point[0])] == 255 and not inv:
                    print(f'Excepetion in {eName} detected!')
                    print(point)
                    M = eDict['matrix']
                else:
                    M = self.default
            except IndexError:
                print('Indexing Error, using default matrix', point)
                M = self.default
        if inv:
            M = np.linalg.inv(M)
        pt = cv2.perspectiveTransform(np.float32([[point]]), M)
        pt = [pt[0][0][0], pt[0][0][1]]
        return pt

if __name__ == '__main__':
    transformer = Transformer('C', True)
    # camera_Vec = a.calculateCameraVec()
    # print(camera_Vec)
    # print(a.transform((100, 200)))
    transformer.saveFile()