import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)#creating image matrix and rgb colors stating type of value
img[:]= 255,255,255 #: means apply to all(this changes color from black to blue)(BGR form)

cv2.circle(img,(256,256),150,(0,69,255),cv2.FILLED)#(image function,(centre point),radius,orange,fil(cv2.FILLED)/thickness(5))
cv2.rectangle(img,(130,226),(382,286),(255,255,255),5) #(image function,(left bottom),(right bottom),color,fil(cv2.FILLED)/thickness(5))
cv2.line(img,(130,296), (382,296),(255,255,255),2)

cv2.putText(img, "Noctis", (137,262),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0),2)


cv2.imshow("Image",img)
cv2.waitKey(0)