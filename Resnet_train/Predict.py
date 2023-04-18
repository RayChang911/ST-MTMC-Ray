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
file_result = open("./REID_car/TestResult_REID_car", 'w')

files = ["./REID_car/Test/car1", "./REID_car/Test/car2",
         "./REID_car/Test/car3", "./REID_car/Test/car4",
         "./REID_car/Test/car5", "./REID_car/Test/car6",
         "./REID_car/Test/car7", "./REID_car/Test/car8",
         "./REID_car/Test/car9", "./REID_car/Test/car10",
         "./REID_car/Test/car11", "./REID_car/Test/car12",
         "./REID_car/Test/car13", "./REID_car/Test/car14",
         "./REID_car/Test/car15", "./REID_car/Test/car16",
         "./REID_car/Test/car17", "./REID_car/Test/car18",
         "./REID_car/Test/car19", "./REID_car/Test/car20",
         "./REID_car/Test/car21", "./REID_car/Test/car22",
         "./REID_car/Test/car23", "./REID_car/Test/car24",
         "./REID_car/Test/car25", "./REID_car/Test/car26",
         "./REID_car/Test/car27", "./REID_car/Test/car28",
         "./REID_car/Test/car29", "./REID_car/Test/car30",
         "./REID_car/Test/car31", "./REID_car/Test/car32",
         "./REID_car/Test/car33", "./REID_car/Test/car34",
         "./REID_car/Test/car35", "./REID_car/Test/car36",
         "./REID_car/Test/car37", "./REID_car/Test/car38",
         "./REID_car/Test/car39", "./REID_car/Test/car40",
         "./REID_car/Test/car41", "./REID_car/Test/car42",
         "./REID_car/Test/car43", "./REID_car/Test/car44",
         "./REID_car/Test/car45", "./REID_car/Test/car46",
         "./REID_car/Test/car47", "./REID_car/Test/car48",
         "./REID_car/Test/car49", "./REID_car/Test/car50",
         ]

# 載入model
model = load_model("./REID_car/model_resnet50_REID_car.h5")

model.summary()
model = Model(inputs=model.input, outputs=model.layers[len(model.layers) - 3].output)  # 共 178 layer
model.summary()

feature_vecs = [[], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], []]

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
        # print(features)
        file_result.write(f + "\n")
        file_result.write(str(features))
        file_result.write("\n" + "\n")

file_result.close()

array_similarity = [[], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], []]

for i, feature_vec_1 in enumerate(feature_vecs):
    for j, feature_vec_2 in enumerate(feature_vecs):
        sim_sum = 0
        count = 0
        for m in feature_vecs[i]:
            for n in feature_vecs[j]:
                sim = cosine_similarity(m, n)
                if sim == 1.:
                    continue
                sim_sum += sim
                count += 1
        sim_sum = round(float(sim_sum / count), 2)
        print(i)
        print(sim_sum)
        array_similarity[i].append(sim_sum)

df_similarity = pd.DataFrame(array_similarity)


print(df_similarity)
df_similarity.to_csv("./REID_car/similarity_result_REID_car.csv")

'''
feature_vecs = [[], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], [],
                [], [], [], [], [], [], [], [], [], []]

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
        # print(features)
        file_result.write(f + "\n")
        file_result.write(str(features))
        file_result.write("\n" + "\n")

file_result.close()

array_similarity = [[], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], [],
                    [], [], [], [], [], [], [], [], [], []]

for i, feature_vec_1 in enumerate(feature_vecs):
    for j, feature_vec_2 in enumerate(feature_vecs):
        sim_sum = 0
        for m in feature_vecs[i]:
            for n in feature_vecs[j]:
                sim = cosine_similarity(m, n)
                sim_sum += sim
        sim_sum = round(float(sim_sum / 100), 2)
        print(i)
        print(sim_sum)
        array_similarity[i].append(sim_sum)

df_similarity = pd.DataFrame(array_similarity,
                             columns=['person1', "person2", "person3", "person4", "person5", "person6", "person7",
                                      "person8", "person9", "person10",
                                      'person11', "person12", "person13", "person14", "person15", "person16",
                                      "person17",
                                      "person18", "person19", "person20",
                                      'person21', "person22", "person23", "person24", "person25", "person26",
                                      "person27",
                                      "person28", "person29", "person30",
                                      'person31', "person32", "person33", "person34", "person35", "person36",
                                      "person37",
                                      "person38", "person39", "person40",
                                      'person41', "person42", "person43", "person44", "person45", "person46",
                                      "person47",
                                      "person48", "person49", "person50",
                                      'person51', "person52", "person53", "person54", "person55", "person56",
                                      "person57",
                                      "person58", "person59", "person60",
                                      'person61', "person62", "person63", "person64", "person65", "person66",
                                      "person67",
                                      "person68", "person69", "person70",
                                      'person71', "person72", "person73", "person74", "person75", "person76",
                                      "person77",
                                      "person78", "person79", "person80",
                                      'person81', "person82", "person83", "person84", "person85", "person86",
                                      "person87",
                                      "person88", "person89", "person90",
                                      'person91', "person92", "person93", "person94", "person95", "person96",
                                      "person97",
                                      "person98", "person99", "person100"],
                             index=['person1', "person2", "person3", "person4", "person5", "person6", "person7",
                                    "person8", "person9", "person10",
                                    'person11', "person12", "person13", "person14", "person15", "person16", "person17",
                                    "person18", "person19", "person20",
                                    'person21', "person22", "person23", "person24", "person25", "person26", "person27",
                                    "person28", "person29", "person30",
                                    'person31', "person32", "person33", "person34", "person35", "person36", "person37",
                                    "person38", "person39", "person40",
                                    'person41', "person42", "person43", "person44", "person45", "person46", "person47",
                                    "person48", "person49", "person50",
                                    'person51', "person52", "person53", "person54", "person55", "person56", "person57",
                                    "person58", "person59", "person60",
                                    'person61', "person62", "person63", "person64", "person65", "person66", "person67",
                                    "person68", "person69", "person70",
                                    'person71', "person72", "person73", "person74", "person75", "person76", "person77",
                                    "person78", "person79", "person80",
                                    'person81', "person82", "person83", "person84", "person85", "person86", "person87",
                                    "person88", "person89", "person90",
                                    'person91', "person92", "person93", "person94", "person95", "person96", "person97",
                                    "person98", "person99", "person100"]
                             )
print(df_similarity)
df_similarity.to_csv("./Real_people/similarity_result_person_V3.csv")
'''
