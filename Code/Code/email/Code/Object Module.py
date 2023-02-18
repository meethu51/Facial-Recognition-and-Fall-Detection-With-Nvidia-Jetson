"""
Object Detection Module
By Bhuvan Shrivastava
"""
import cv2

#This function finds objects using the haarcascade xml file
def findObjects(img, objectCascade,scaleF = 1.1, minN = 4):

    """
    parameters ~
    img : Image in which the object neds to be found
    objectCascade :Object Cascade created with the Cascade Classifier
    scaleF :How much the image sizeis reduced at each image scaling
    minN : how many neighbours each rectangle should have to accept as valid
    return: Image with the rectangles draw and the bounding box info 
    """
    imgObjects = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = objectCascade.detectMultiScale(imgGray,1.1,4)

    for(x,y,w,h) in objects :
        cv2.rectangle(imgObjects,(x,y),(x+w,y+h),(255,0,255),2)
    
    return imgObjects, objects


#making the main function to be called later
def main():
    img = cv2.imread("Resources/Face.jpeg")
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    imgObjects, objects =  findObjects(img, faceCascade)
    cv2.imshow("Object Detection",imgObjects)
    cv2.waitKey(0)

#making it into a module
if __name__ == "__main__" :
    main()