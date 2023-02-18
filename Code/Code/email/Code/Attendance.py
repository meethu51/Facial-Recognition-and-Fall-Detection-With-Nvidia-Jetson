import cv2
import sqlite3
import numpy as np
import face_recognition
import os
from datetime import datetime
# import mysql.connector
# connection=sqlite3.connect("my_database")
# cursor=connection.cursor()
# table=cursor.execute("""  CREATE TABLE MY_TABLE (ID INTEGER PRIMARY KEY AUTOINCREMENT,
#                         DETECTED_PERSON TEXT NOT NULL, TIMESTAMP TEXT NOT NULL); """)
# connection.commit()
# connection.close()   
path = '/home/bhuvan/Desktop/Code/email/Code/training'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('/home/bhuvan/Desktop/Code/email/Code/attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',') 
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H,%M,%S')
            f.writelines(f'\n{name},{dtstring}')



encodeListKnown = findEncodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)

while True :
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame= face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print("faceDis",faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print("name",name)
            # query="""INSERT INTO MY_TABLE (DETECTED_PERSON, TIMESTAMP) Values(?, CURRENT_TIMESTAMP) """
            # data=[name]
            # cursor.execute(query,data)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    
    if cv2.waitKey(1) & 0XFF == ord ('o') :
        # connection.commit()
        # connection.close()
        break
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)