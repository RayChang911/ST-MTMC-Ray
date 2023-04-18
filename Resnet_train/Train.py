from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.models import Sequential

img_height, img_width = (244, 244)
batch_size = 64
class_num = 3

train_data_dir = "./Classifier_V3/Processed_data/train"
valid_data_dir = "./Classifier_V3/Processed_data/val"
test_data_dir = "./Classifier_V3/Processed_data/test"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')
valid_generator = train_datagen.flow_from_directory(valid_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')
test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                   target_size=(img_height, img_width),
                                                   batch_size=1,
                                                   class_mode='categorical')

x, y = test_generator.next()
print(x.shape)
''' 
# ResNet50 For REID
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                      input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling layer
x = (Dropout(0.1))(x)
x = (Flatten())(x)
x = (Dropout(0.25))(x)
x = Dense(1024, activation='relu')(x)  # fully connected layer having 1024 neurons
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
'''
# Simple Model For Classifier
model_ = Sequential()
model_.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(244, 244, 3)))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model_.add(MaxPooling2D(pool_size=(2, 2)))
model_.add(Dropout(0.1))
model_.add(Flatten())
model_.add(Dropout(0.25))
model_.add(Dense(class_num, activation='softmax'))

model_.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_.summary()

model_.fit(train_generator,
           batch_size=batch_size,
           validation_data=valid_generator,
           epochs=30,
           verbose=1)
model_.save("./Classifier_V3/model_resnet50_Classifier_V3.h5")
# model = tf.keras.models.load_model("D:/Ray/Resnet_train/processed_data/model-resnet50-final.h5")
# print(model.summary())

#######################################################
# test

test_loss, test_acc, = model_.evaluate(test_generator, verbose=1)
print("Test Loss: " + str(test_loss))
print("Test Accuracy: " + str(test_acc))
