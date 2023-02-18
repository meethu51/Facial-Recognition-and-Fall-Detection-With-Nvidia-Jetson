import cv2

vid = cv2.VideoCapture(0)

while (True) :

    ret, image = vid.read()
    cv2.imshow('image', image)
    
    if cv2.waitKey(1) & 0XFF == ord ('q') :
        break

    cv2.imwrite('/home/bhuvan/Desktop/Code/email/Code/training//imagex.jpg', image)

vid.release()
cv2.destroyAllWindows()