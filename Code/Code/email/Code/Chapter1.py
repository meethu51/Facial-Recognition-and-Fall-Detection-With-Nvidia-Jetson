import cv2
print(cv2.__version__)


#import an image
img = cv2.imread("Resources/images.jpeg")
cv2.imshow("output", img)
cv2.waitKey(0)

#import video
frameWidth =640
frameHeight = 480
cap = cv2.VideoCapture("Resources/10 Seconds of_ Cat.mp4")

while True:
        sucess, img = cap.read()
        img = cv2.resize (img, (frameWidth,frameHeight))
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

#Run Webcam
frameWidth =1280
frameHeight = 720
cap = cv2.VideoCapture(0)

while True:
        sucess, img = cap.read()
        img = cv2.resize (img, (frameWidth,frameHeight))
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break