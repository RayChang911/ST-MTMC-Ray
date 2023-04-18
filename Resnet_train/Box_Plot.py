import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

confusion_matrix = []
with open("./REID_motorcycle/similarity_result_REID_motorcycleForBoxPlot.csv", newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        confusion_matrix.append(row)

confusion_arr = np.array(confusion_matrix)
print(confusion_arr)
confusion_arr_dia = []
confusion_arr_oth = []
for idx, x in np.ndenumerate(confusion_arr):
    if idx[1] == idx[0]:
        confusion_arr_dia.append(float(x))
    elif idx[1] > idx[0]:
        confusion_arr_oth.append(float(x))

df_dia = pd.DataFrame(confusion_arr_dia)
print("confusion_arr_dia")
print(df_dia.describe())

df_oth = pd.DataFrame(confusion_arr_oth)
print("confusion_arr_oth")
print(df_oth.describe())

plt.title('Resnet50 REID Motorcycle Boxplot')
labels = "The same object", "Not the same object"
plt.ylabel('Cosine Similarity')
plt.boxplot([confusion_arr_dia, confusion_arr_oth], labels=labels, showmeans=True, showfliers=False)
plt.show()  # 顯示圖像
