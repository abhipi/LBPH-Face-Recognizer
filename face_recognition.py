import cv2 as cv
import numpy as np
from PIL import Image
import os

path="C:/Users/Dell/Desktop/face detection and recognition/dataset" #location of your dataset

recognizer=cv.face.LBPHFaceRecognizer_create()
detector=cv.CascadeClassifier("C:/Users/Dell/Desktop/face detection and recognition/haarcascade_frontalface_default.xml")#location of the haar cascade xml file

def getImageAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]

    for imagePath in imagePaths:
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')
        id=int(os.path.split(imagePath)[-1].split("_")[1])
        faces=detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids



print("\n Training faces")
faces,ids=getImageAndLabels(path)

recognizer.train(faces,np.array(ids))

recognizer.write("C:/Users/Dell/Desktop/face detection and recognition/trainer/trainer.yml")

print("Model is trained on "+str(len(np.unique(ids)))+" no of faces")
