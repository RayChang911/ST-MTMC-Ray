from keras.applications.resnet import preprocess_input
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.preprocessing import image
import numpy as np


model = load_model("./Classifier_V3/model_resnet50_Classifier_V3.h5")

model = Model(inputs=model.input, outputs=model.output)  # 共 178 layer
model.summary()

f_mor = "./Classifier_V3/Test/motorcycle/CCTV1659.jpg"
f_per = "./Classifier_V3/Test/person/0204_c3s1_040726_04.jpg"

img = image.load_img(f_mor, target_size=(244, 244, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
features = features.flatten().reshape(1, -1)
print(features)
print(np.argmax(features))

img = image.load_img(f_per, target_size=(244, 244, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
features = features.flatten().reshape(1, -1)
print(features)
print(np.argmax(features))


