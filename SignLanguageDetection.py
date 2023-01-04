#Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector #For detecting
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
#import tensorflow
import time
import pyttsx3 as tts

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

#Importing trained model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 400
count = 0

#Text to Speech
eng = tts.init()
speed = eng.getProperty('rate')
eng.setProperty('rate', 160)
eng.say("Hello! Nice to meet you. You can now start by showing your hand")
print("Show your hands")
eng.runAndWait()
time.sleep(3)
Labels = ["A", "B", "C", "D", "E", "DISLIKE", "F", "G", "H", "I",  "K", "L", "M", "N", "O", "OKAY", "P", "Q", "R", "S", "ROCK", "T", "U", "V", "W", "X", "Y"]

while True:
    time.sleep(1)
    success, img = cap.read()
    OutputImg = img.copy()
    hands, img = detector.findHands(img)

    #Cropping the image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        #Centralizing the image
        asp_ratio = h/w
        if asp_ratio > 1:
            i = imgSize/h
            newWidth = math.ceil(i*w)
            ResizeImg = cv2.resize(imgCrop, (newWidth, imgSize))
            imgReShape = ResizeImg.shape
            WidthGap = math.ceil((imgSize-newWidth)/2)
            imgWhite[:, WidthGap:(newWidth+WidthGap)] = ResizeImg
            predict, index = classifier.getPrediction(imgWhite, draw=False)
            print(predict, index)
        else:
            i = imgSize / w
            newHeight = math.ceil(i * h)
            ResizeImg = cv2.resize(imgCrop, (imgSize, newHeight))
            imgReShape = ResizeImg.shape
            HeightGap = math.ceil((imgSize - newHeight) / 2)
            imgWhite[HeightGap:newHeight + HeightGap, :] = ResizeImg
            predict, index = classifier.getPrediction(imgWhite, draw=False)
            print(predict, index)

        cv2.rectangle(OutputImg, (x-offset, y-offset-50), (x-offset+200, y-offset), (255, 155, 255), cv2.FILLED)
        cv2.putText(OutputImg, Labels[index], (x, y-24), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 1, 255), 2)
        cv2.rectangle(OutputImg, (x-offset, y-offset), (x+w+offset, y+w+offset), (255, 155, 255), 4)
        cv2.imshow("ImageWhites", imgWhite)
        cv2.imshow("ImageCrop", imgCrop)

    #Output-prediction
    cv2.imshow("Image", OutputImg)
    eng.say(Labels[index])
    eng.runAndWait()
    cv2.waitKey(1)