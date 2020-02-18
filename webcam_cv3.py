import cv2
import sys
import glob
import os
import logging as log
import datetime as dt
import xlrd
import pandas as pd
import numpy as np
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    # get the latest cvs file
    list_of_files = glob.glob('/Users/xiaojun/Work/DEMO/body_temperature/temp_folder/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    # to open the file
    print(latest_file)
    
    df = pd.read_csv(latest_file)
    df = df.values

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        df = pd.read_csv(latest_file)
        df = df.values
        spot_tem = df[121,1]
        print(spot_tem)
        xx = int(160/width*x)
        yy = int(120/height*y)
        ww = int(160/width*w)
        hh = int(120/height*h)
        dff = df[xx:xx+ww, yy:yy+hh]
        mmax = np.amax(dff)
        print(mmax + spot_tem - 23.0)
        cv2.putText(frame, str(mmax + spot_tem -23.0), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
   
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
