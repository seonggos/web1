#!Python

import sys
import numpy as np
import cv2
import os
import tensorflow as tf
from random import shuffle
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


test_data = "./uploads"
result_list = []

def test_data_with_label():
	test_images = []

	for i in os.listdir(test_data):
		path = os.path.join(test_data, i)
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (224, 224))
		test_images.append([np.array(img), i])
	return test_images

testing_images = test_data_with_label()
model = Sequential()
model.add(InputLayer(input_shape = [224, 224, 3]))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=4, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=4, padding='same'))
model.add(Conv2D(filters=96, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=4, padding='same'))
model.add(Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=4, padding='same'))
model.add(Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu'))


model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=1e-3)
model.load_weights("model.h5")
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

result = {
    "P-p" : 0,
    "P-n" : 0,
    "N-p" : 0,
    "N-n" : 0
}

img_num = 1

for data in testing_images:
    img = data[0]
    title = data[1]
    data = img.reshape(1, 224, 224, 3)
    model_out = model.predict([data])
    print(str((round(((model_out[0][1])*100), 1)))+"%")
    # print("picture " + str(img_num) + ": " + str((round(((model_out[0][1])*100), 1)))+"%")
    img_num += 1

    if np.argmax(model_out) == 1:
        str_label = "Negative"
    else:
        str_label = "Positive"

    if str_label == "Positive" and title.split("_")[0] == "positive":
        result["P-p"] += 1;
    elif str_label == "Positive" and title.split("_")[0] == "negative":
        result["P-n"] += 1;
    elif str_label == "Negative" and title.split("_")[0] == "positive":
        result["N-p"] += 1;
    elif str_label == "Negative" and title.split("_")[0] == "negative":
        result["N-n"] += 1;



result_message = "There are " + str(result["N-p"] + result["N-n"]) + " negative pictures of " + str(len(testing_images))
# result_message = ("" + str(result_list) + " There are " + str(result["N-p"] + result["N-n"]) + " negative pictures of " + str(len(testing_images)))


print("content-type: text/html; charset=utf-8\n")
print(result_message)
# print(len(testing_images))
