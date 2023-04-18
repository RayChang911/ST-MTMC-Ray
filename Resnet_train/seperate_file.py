import shutil
from os import walk
from os.path import join

MyPath  = './Market-1501-v15.09.15/gt_bbox/' # 當下目錄
n = 784
numder = 1


for root, dirs, files in walk(MyPath):
  for i in files:
    FullPath = join(root, i) # 獲取檔案完整路徑
    FileName = join(i) # 獲取檔案名稱
    print(FullPath)
    filenum = int(FileName[0:4])
    print(filenum)
    if(filenum == numder):
      shutil.move(FullPath, './ResNet_Dataset_MRT/traindata/'+str(n))
    else:
      n+=1
      numder = filenum
      shutil.move(FullPath, './ResNet_Dataset_MRT/traindata/' + str(n))
