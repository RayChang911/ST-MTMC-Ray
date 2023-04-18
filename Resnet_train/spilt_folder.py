import splitfolders

input_f = "./REID_motorcycle/Train_data"
output_f = "./REID_motorcycle/Processed_data"
splitfolders.ratio(input_f, output_f, seed=1337, ratio=(.7, .2, .1))

help(splitfolders.ratio)
