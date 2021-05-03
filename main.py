import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

path = 'imagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
myList.pop(0)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList :
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.face_distance(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            if matches[matchIndex] <= 0.6:
                name = classNames[matchIndex].upper()
                colorGrad = (0, 255, 0)
                textSize = (x1+6, y2-6)
                # cv2.imshow("Original img", img)
            else:
                name = "NOMALUM SHAXS!"
                colorGrad = (50, 0, 255)
                textSize = (x1+6, y2-6)
            # print(name)
            # print(matches)
            cv2.rectangle(img, (x1, y1), (x2, y2), colorGrad, 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), colorGrad, cv2.FILLED)
            cv2.putText(img, name, textSize,cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),2)
            markAttendence(name)

    df = pd.read_csv('Attendence.csv')
    writer = pd.ExcelWriter('test.xlsx')
    df.to_excel(writer, index=False)
    writer.save()

    cv2.imshow('Next-Gen', img)
    cv2.waitKey(1)