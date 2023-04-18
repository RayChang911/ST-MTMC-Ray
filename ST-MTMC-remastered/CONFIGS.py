DATASET = 'Real'

# ROOT_PATH = f'./datasets/{DATASET}'
ROOT_PATH = f'D:/Ray/ST-MTMC-remastered/datasets/{DATASET}'
OUT_PATH = f'./results/{DATASET}'

CAMERAS_PATH = f'{ROOT_PATH}/cameras'
VIDEOS_PATH = f'{ROOT_PATH}/videos'
FLOOR_PATH = f'{ROOT_PATH}/FloorPlan.jpg'

'''
# For Unity
CAMERAS_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
STREAM_SIZE = (1280, 720)
'''

# For Real
CAMERAS_LIST = ['A', 'B', 'C']
STREAM_SIZE = (1920, 1080)


STREAM_FPS = 30
PROCESS_FPS= 1000

OBJLOG_FILE = f'{OUT_PATH}/obj_log.json'
GROUPS_FILE = f'{OUT_PATH}/groups.json'
TRACKLETS_PATH = f'{OUT_PATH}/tracklets/'