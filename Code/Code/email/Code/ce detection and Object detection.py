import cv2

img = cv2.imread("Resources/people.jpg")
faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml") 
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for(x,y,w,h) in faces :
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

cv2.imshow("Object Detection",img)
cv2.waitKey(0)