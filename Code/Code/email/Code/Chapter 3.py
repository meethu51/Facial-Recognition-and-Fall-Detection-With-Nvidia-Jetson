import cv2
import numpy as np
 
img = cv2.imread("Resources/images.jpeg")
print(img.shape)#tells u dimension wtih color

imgResize = cv2.resize(img, (640,480))#By Size
imgResize1 = cv2.resize(img, (0,0), None, 4.0,4.0)#by Scale
imgCropped = img[100:200,200:300]

cv2.imshow("Image", img)
cv2.imshow("image Resize", imgResize)
cv2.imshow("image Resize1", imgResize1)
cv2.imshow("image Cropped", imgCropped)
cv2.waitKey(0)