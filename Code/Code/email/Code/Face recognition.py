import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np
# import pyodbc

path = "/home/bhuvan/Desktop/Code/email/Code/training"
images = []
imgLabel  = []
myList = os.listdir(path)

for currentlist in myList :
    currentImg = cv2.imread(f'{path}\\ {currentlist}')
    images.append(currentImg)
    imgLabel.append(os.path.splitext(currentlist)[0])

def findEncodings(images):
    encodeList=[]
    for img in images :
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodelistKnownFaces = findEncodings(images)
print('Encoding Complete')

# def markAttendance2(name,InTime,InDate):
#     conn = pyodbc.connect('Driver = {SQL Server};'
#                           'SERVER = localhost:3306;'
#                           'Database = Facial Recognition'
#                           'Trusted_Connection = yes')

#     cursor = conn.cursor()

#     sql = '''insert into Facial Recogntion.dbo.tbl_Facial recognition 1 (Name,InDate.InTime) values(?,?,?)'''

#     val = (name,InDate,InTime)
#     cursor.execute(sql, val)
#     conn.commit()



webcam = cv2.VideoCapture(0)
AttendanceValidate = 'a' 

while True :
    success, img  = webcam.read()
    imgS = cv2.resize(img,(0,0), None , 0.25,0,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)

    for encodeFace, facelocation in zip(encodeCurrentFrame,faceCurrentFrame):
        match = face_recognition.compare_faces(encodelistKnownFaces, encodeFace)
        faceDistance = face_recognition.face_distance(encodelistKnownFaces, encodeFace)

        matchIndex = np.argmin(faceDistance)

        if match[matchIndex]:
            name =  imgLabel[matchIndex].upper()
            y1,x2,y2,x1 = facelocation
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3) 
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            crTime = datetime.now().time()
            crDate = datetime.now().date()
            if name!= AttendanceValidate:
                #markAttendance2(name,str(crTime),str(crDate))
                AttendanceValidate = name

    cv2.imshow('Frame' , img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()



