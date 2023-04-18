from utils.mAHT import *
from CONFIGS import CAMERAS_LIST
for cam in CAMERAS_LIST:
    print(f'Setting cam {cam}')
    transformer = Transformer(cam, True)
    if input('Saving?') in ['y', 'Y']:
        transformer.saveFile()
    