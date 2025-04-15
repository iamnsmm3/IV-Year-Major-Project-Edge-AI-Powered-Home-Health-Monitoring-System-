
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import socket
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle
from keras.models import load_model
import cv2
import numpy as np

main = tkinter.Tk()
main.title("A Deep Transfer Learning-based Edge Computing Method for Home Health Monitoring") #designing main screen
main.geometry("1300x1200")

global filename
global model
global X,Y

class_labels = ['Fall', 'No Fall']

def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
def loadModel():
    text.delete('1.0', END)
    global model
    model = load_model('model/model.h5')
    text.delete('1.0', END)
    text.insert(END,"VGG16 Deep Transfer Learning Health Monitoring Model Loaded\n\n");

def sendToCloud(img, result):
    img = cv2.resize(img,(100,100))
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect(('localhost', 2222))
    features = []
    features.append("sensordata")
    features.append(result)
    features.append(img)
    features = pickle.dumps(features)
    client.send(features)

def fallDetection():
    global model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    temps = cv2.imread(filename)
    img_h, img_w, c = temps.shape
    image = load_img(filename, target_size=(80, 80))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    (boxPreds, labelPreds) = model.predict(image)
    boxPreds = boxPreds[0]
    x = boxPreds[0]
    y = boxPreds[0]
    w = boxPreds[0]
    h = boxPreds[0]
    x_min = int((x-w/2)*img_w)
    y_min = int((y-h/2)*img_h)
    x_max = int((x+w/2)*img_w)
    y_max = int((y+h/2)*img_h)
    cv2.rectangle(temps, (x_min,y_min), (x_max, y_max), (0, 255, 0), 2)
    predict= np.argmax(labelPreds, axis=1)
    predict = predict[0]
    accuracy = np.amax(labelPreds, axis=1)
    accuracy = accuracy[0]
    print(str(class_labels[predict])+" "+str(accuracy))
    temps = cv2.resize(temps, (600,500))
    cv2.putText(temps, "Health Condition Predicted As "+str(class_labels[predict]), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(temps, "Prediction Accuracy "+str(accuracy), (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    sendToCloud(temps, class_labels[predict])
    cv2.imshow("Health Condition Predicted As "+str(class_labels[predict]), temps)
    cv2.waitKey(0)

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='A Deep Transfer Learning-based Edge Computing Method for Home Health Monitoring')
title.config(bg='deep sky blue', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Fall Detection Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

loadButton = Button(main, text="Generate & Load Health Monitoring Model", command=loadModel)
loadButton.place(x=350,y=550)
loadButton.config(font=font1) 

detectionButton = Button(main, text="Detect Fall Detection from RGB Image", command=fallDetection)
detectionButton.place(x=740,y=550)
detectionButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=600)
exitButton.config(font=font1) 


main.config(bg='LightSteelBlue3')
main.mainloop()
