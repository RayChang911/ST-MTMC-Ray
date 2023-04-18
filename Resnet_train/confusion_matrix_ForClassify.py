import pandas as pd
import tensorflow as tf
import torch
from keras.applications.resnet import preprocess_input
from keras.layers import Dense, Dropout, Lambda
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# 寫txt檔
# file_result = open("./Classifier_V3/TestResult.txt", 'w')

files = ["./Classifier_V3/Test/car", "./Classifier_V3/Test/motorcycle", "./Classifier_V3/Test/person"]

# files = "./Real_people/Test"

# 載入model
model = load_model("./Classifier_V3/model_resnet50_Classifier_V3.h5")

model = Model(inputs=model.input, outputs=model.output)  # 共 178 layer
# model.summary()

feature_vecs = [[], [], []]

# 讀取資料夾內的檔案
for i, file in enumerate(files):
    for filename in os.listdir(file):
        f = os.path.join(file, filename)
        img = image.load_img(f, target_size=(244, 244, 3))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        features = features.flatten().reshape(1, -1)
        feature_vecs[i].append(features)

        # print(f)
        print(features)
        # file_result.write(f + "\n")
        # file_result.write(str(features))
        # file_result.write("\n" + "\n")

# file_result.close()

array_similarity = [[[], [], []], [[], [], []], [[], [], []]]
# print(len(feature_vecs))

for i, feature_vec_arr in enumerate(feature_vecs):
    match_num_0 = 0
    match_num_1 = 0
    match_num_2 = 0
    total_num = 0
    for j, feature_vec in enumerate(feature_vec_arr):
        print(j)
        pre = np.argmax(feature_vec)
        print(pre)
        # if feature_vec[pre] < 0.75:
            # continue

        if pre == 0:
            match_num_0 += 1
        elif pre == 1:
            match_num_1 += 1
        elif pre == 2:
            match_num_2 += 1
        total_num += 1
    match_rate_0 = round(match_num_0 / total_num, 5)
    match_rate_1 = round(match_num_1 / total_num, 5)
    match_rate_2 = round(match_num_2 / total_num, 5)
    array_similarity[i][0] = match_rate_0
    array_similarity[i][1] = match_rate_1
    array_similarity[i][2] = match_rate_2

print(array_similarity)

df_similarity = pd.DataFrame(array_similarity,
                             columns=['car', "motorcycle", "person"],
                             index=['car', "motorcycle", "person"]
                             )
print(df_similarity)
df_similarity.to_csv("./Classifier_V3/similarity_result.csv")
