import cv2
import numpy as np

img = cv2.imread("Resources/images.jpeg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),6)
imgCannay = cv2.Canny(imgBlur, 100, 150)#outlines
kernel = np.ones((5,5),np.uint8) #thickness amount
imgDia = cv2.dilate(imgCannay, kernel,iterations=1) #thickness iterations tell u how many tims the numpy is applied
imgErode = cv2.erode(imgDia, kernel,iterations=1)


cv2.imshow("Output",img)
cv2.imshow("ImageGray",imgGray)
cv2.imshow("ImageBlur",imgBlur)
cv2.imshow("Image Canny",imgCannay)
cv2.imshow("Image Dia",imgDia)
cv2.imshow("Image Erode",imgErode)
cv2.waitKey(0)