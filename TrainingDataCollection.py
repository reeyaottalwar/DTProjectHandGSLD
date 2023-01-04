#Libraries
import cv2
from cvzone.HandTrackingModule import HandDetector #For detecting
import numpy as np
import math
import time
import pyttsx3 as tts #Text to speech

#Main
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 400
count = 0

#Text to Speech
eng = tts.init()
RateOS = eng.getProperty('rate')
eng.setProperty('rate', RateOS-25)
eng.say("Hello")
eng.runAndWait()
time.sleep(3)
print("\...You can show your hand sign and press \'s\' to save that image...")

#Data Storing File
folder = "venv\Data\A"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    #Cropping the image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        #x1, y1, w1, h1 = hand['vbox']
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Box
        #Centralizing the image
        asp_ratio = h/w
        if asp_ratio > 1:
            i = imgSize/ h
            newWidth = math.ceil(i*w)
            ResizeImg = cv2.resize(imgCrop, (newWidth, imgSize))
            imgReShape = ResizeImg.shape
            WidthGap = math.ceil((imgSize-newWidth)/2)
            imgWhite[:, WidthGap:(newWidth+WidthGap)] = ResizeImg
        else:
            i = imgSize / w
            newHeight = math.ceil(i * h)
            ResizeImg = cv2.resize(imgCrop, (imgSize, newHeight))
            imgReShape = ResizeImg.shape
            HeightGap = math.ceil((imgSize - newHeight) / 2)
            imgWhite[HeightGap:newHeight + HeightGap, :] = ResizeImg

        cv2.imshow("ImageWhites", imgWhite)
        cv2.imshow("ImageCrop", imgCrop)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    #Saving the images
    if key == ord("s"):
        count += 1
        cv2.VideoWriter(f'{folder}/Img_{time.time()}.mp4', imgWhite)
        print("Counter: ", count)