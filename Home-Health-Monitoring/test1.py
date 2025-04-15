import xml.etree.ElementTree as ET
import os
from keras.preprocessing.image import load_img
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import pickle
import cv2

from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
import pandas as pd
'''
dataset = 'Dataset/labels'

data = []
labels = []
bboxes = []
size = []

for root, dirs, directory in os.walk(dataset):
    for j in range(len(directory)):
        dataset = pd.read_csv(root+'/'+directory[j],header=None,sep=" ")
        dataset = dataset.values
        x = dataset[0,1]
        y = dataset[0,2]
        w = dataset[0,3]
        h = dataset[0,4]
        name = directory[j]
        name = name.replace(".txt",".jpg")
        image = load_img("Dataset/train/"+name, target_size=(80, 80))
        lbl = 0
        if "fall" in name:
            labels.append(0)
            lbl = 0
        if "not" in name:
            labels.append(1)
            lbl = 1
        bboxes.append([x, y, w, h])
        image = img_to_array(image)
        data.append(image)
        img = cv2.imread("Dataset/train/"+name)
        height, width, c = img.shape
        size.append([width,height])
        print(str([x, y, w, h])+" "+name+" "+name+" "+str(lbl))
        
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes)
size = np.array(size)

labels = to_categorical(labels)
print(labels)
#print(lb.classes_)

np.save('model/img.txt',data)
np.save('model/labels.txt',labels)
np.save('model/bbox.txt',bboxes)
np.save('model/size.txt',size)
'''
data = np.load('model/img.txt.npy')
labels = np.load('model/labels.txt.npy')
bboxes = np.load('model/bbox.txt.npy')
size = np.load('model/size.txt.npy')
print(labels[0])
print(labels.shape)


img = data[0]
image_size = size[0]
img_h = image_size[1]
img_w = image_size[0]
img = cv2.resize(img,(img_w,img_h))
bb = bboxes[0]
print(bb)
x_min = int((bb[0]-bb[2]/2)*img_w)
y_min = int((bb[1]-bb[3]/2)*img_h)
x_max = int((bb[0]+bb[2]/2)*img_w)
y_max = int((bb[1]+bb[3]/2)*img_h)

#img = cv2.resize(img,(200,200))
cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0, 255, 0), 2)
cv2.imshow("Output", img)
cv2.waitKey(0)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
bboxes = bboxes[indices]
size = size[indices]
print(bboxes.shape)
print(data.shape)
print(labels.shape)


split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
#(trainPaths, testPaths) = split[6:]

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(80, 80, 3)))
vgg.trainable = False
flatten = vgg.output
flatten = Flatten()(flatten)


bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}
opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}
hist = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=25, verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save('model/model.h5')
f = open('model/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()


