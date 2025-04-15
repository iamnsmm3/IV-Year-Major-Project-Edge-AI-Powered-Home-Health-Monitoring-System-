import pandas as pd
import numpy as np
import cv2

dataset = pd.read_csv("Dataset/labels/fall001.txt",header=None,sep=" ")
dataset = dataset.values
x = dataset[0,1]
y = dataset[0,2]
w = dataset[0,3]
h = dataset[0,4]


name = "fall001.txt"
name = name.replace(".txt",".jpg")

img = cv2.imread("Dataset/train/"+name)
img_h, img_w, c = img.shape

x_min = int((x-w/2)*img_w)
y_min = int((y-h/2)*img_h)
x_max = int((x+w/2)*img_w)
y_max = int((y+h/2)*img_h)

cv2.rectangle(img, (x_min,y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.imshow("test",img)
cv2.waitKey(0)

