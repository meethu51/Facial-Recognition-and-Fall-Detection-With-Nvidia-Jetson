# Facial-Recognition-and-Fall-Detection-With-Nvidia-Jetson
Abstract  

The goal of this project is to create a program that can perform facial recognition and fall detection using the Jetson Xavier AGX

Main Software Required:

1.Visual Studio Code
2.Python
3. Balena Etcher
4. SD Card formatter
5. Python libraries
Cmake 
Tensorflow
Dlib
Swig
Numpy
Tf-pose-estimation library
Tkinter
CustomTkinter
Git
OpenCV
face_recognition
6. Linux

Main Hardware Required:

1. Nvidia Jetson Nano/Nvidia Jetson Xavier/Nvidia Orin


2. A USB Webcam (eg:Logitech Webcam C270)


3. USB to SD Card Reader
4. At least a 32GB SD card



Table of Contents



Abstract
Acknowledgement
Table of Contents
Introduction
1.1 Background Information
Project objective(s)
Project Scope
3.1 Program Flow
Hardware Development
4.1 Setting up the Nvidia Jetson Device
Software Development
5.1 Setting up the environment
5.2 Installing Libraries
5.3 Finding Resources 
 Capturing Faces
 Facial Recognition
7.1 Mini Database for Fall Recognition data
7.2 Training Dataset for Facial Recognition
 Fall Detection
8.1 Training dataset for fall Detection
8.2 Creating the program
Problems Encountered
Gantt Chart
Conclusion
Appendix

























Introduction

1.1 Background Information

Our purpose of this project was to create a user friendly software that would help monitor patients and elderly at healthcare centres like hospitals and elderly homes. Instead of always looking after the patients the program would be able to detect if the correct number of people are there, if anyone who is not in the system is there as well as if someone has fallen down. The timing of both incident as to when a new person has entered the field of vision and when someone has fallen are both recorded as well. When a person falls down a notification will pop up which will allow the nearest caretake to help the patient or elderly. 









Project objective(s)

To design a user friendly software with a graphical user interface that can perform activities of the follow nature : Face Capture, Facial Recognition, Fall Detection.

Image Example of Facial Recognition

Image Example of Fall Detection 





Project Scope

3.1 Program Flow

This project requires the student to create a program that can perform face capture, facial recognition and fall detection using python libraries to train a database and detect live footage. Additionally data will be recorded onto an excel file. The following flow charts show how fall recognition and facial recognition is performed. Ample knowledge of programming, python and machine learning is recommended in this process.





When performing facial recognition we will be using the concept of neural networks to form connections. I have displayed it in the format of photos for easier understanding.







Hardware Development

4.1 Setting up the Nvidia Jetson Device

We will be using either the Jetson nano or Jetson Xavier AGX for this project as they are the best hardware choices for doing a machine learning project.

We will begin by downloading the Jetpack operating system from the Nvidia Website, Balena Etcher and SD Card Formatter here
1.https://developer.nvidia.com/embedded/jetpack
2.https://www.balena.io/etcher
3.https://www.sdcard.org/downloads/formatter/

Install Balena etchera and SD Card formatter .Connect the SD card to your Computer using the SD card reader and open the SD card Formatter and format your card using that software. Use balena etcher to select the jetpack file and etch onto your SD Card.

Once this is completed Inserted SD card to the bottom of the Jetson Nano board.
For the Jetson Xavier AGX you can direly connect the device to your computer and install it onto the 32GB Storage inside the Device.

Note : for the Jetson nano you should use a jumper cable to use the proper adapter for utmost efficiency 

Upon turning on the Jetson Nano upon connecting it to a monitor you will be met with a linux Installation screen. Proceed to click next until the installation is completed.

You will be able to Accces the Jetson Ubuntu environment.For an easier time you can follow this video guide by Nvidia

Jetson Nano: https://www.youtube.com/watch?v=uvU8AXY1170&ab_channel=NVIDIADeveloper
Jetson Xavier : https://www.youtube.com/watch?v=-nX8eD7FusQ

Software Development

5.1 Setting up the environment

In order to make sure our program runs smoothly we will first need to install Visual studio code and python version 3.9

To do this just go onto the command line and type the following.

Sudo apt-get update 
Sudo apt-get upgrade

sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update 
sudo apt install python3.9
It is a good rule of thumb to sudo apt-get update and sudo apt-get upgrade everyday to ensure all the latest dependencies are installed.

5.2 Installing Libraries

Firstly we will have to start by installing these libraries
Cmake 
OpenCV
Tensorflow
Dlib
Swig
Numpy
Tensorflow
Tf-pose-estimation library
Tkinter
CustomTkinter
Git
face_recognition
And many more

You can do this by going to this website searching each library and following its instructions.
https://pypi.org/

However to install tensorflow please follow this website provided by Nvidia as it is the most suitable one.
https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

For Cmake you might want to follow this website. However incase it is taken down in the future I have attached the code here.

https://bigphongsakhon.medium.com/human-pose-estimation-using-open-pose-on-jetson-nano-28b964c2f0c2

TO INSTALL CMAKE
apt-get update
apt-get install -y libssl-dev libcurl4-openssl-dev qt5-default
apt-get install -y build-essential libboost-all-dev libboost-dev libhdf5-dev libatlas-base-dev
apt-get install -y python3-dev python3-pip
apt-get remove -y cmake
cd /usr/local/src
wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz
tar -xvzf cmake-3.19.4.tar.gz
cd cmake-3.19.4
./bootstrap --qt-gui
make -j4
make install
apt-get install -y libprotobuf-dev protobuf-compiler libgflags-dev libgoogle-glog-dev

After this open cmake using the search bar and do the following steps

1.Where the source code: Enter Directory folder of tf-pose-estimation 
2.Where to build the binaries: Enter the Directory of Folder build which is located in Folder  tf-pose-estimation.
3. Press Configure. If a new window pops up, select “Unix Makefile” and press OK.
4.BUILD_PYTHON: tick the check mark
5.USE_CUDNN: Remove the check mark.
6. Press Generate

5.3 Finding Resources

If at any point in the process you are stuck and require assistance you can go to my github repository or google the errors to look for an answer. You will face numerous challenges when attempting this project some which i might not have faced. So google and youtube are your best friends.












Capturing Faces

Once this has been completed you will need to write this code for face capture
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
Note : the directory can be changed to that of your preferences as to where you want to store the images. 

In order to take a picture just press q on your keyboard to take the photo and save it

Now you can perform face capture and take photos.










Facial Recognition

We can now proceed to the next portion of how do facial recognition. But firstly what is facial recognition? 

A facial recognition system is a technology capable of matching a human face from a digital image or a video frame against a database of faces. 



How does it find a face?


In conjunction with OpenCV and Dlib, the face_recognition library looks for the above highlighted features like mouth, eyes, eyebrows, nose and chin to draw a conclusion that there is a face in the photo or video feed.


More Technically it works like this :

It works by Identifying and Measuring facial Features in an image using (x,y) coordinate system
1.Image is Captured
2.Eye Locations are determined.
3.Image is converted to gray scale and cropped
4. Image is converted into a template for comparing facial recognition results.
5. Matched image as the result.























Training a dataset 

Firstly we will have to train a dataset


This is the flow of how to train a dataset

To train a dataset we will need to run the code in this video on our system. If you believe your jetson device is not powerful enough you can alternatively run the code on a jupyter notebook by following this video guide I found on youtube.

How To Train Deep Learning Models In Google Colab- Must For Everyone

Do remember to change to path directories of your image dataset. If you would like to find a face dataset you can get one of your choice from google. I would recommend getting one from Kaggle as thats where I got mine from.



However if you would like a pretrained dataset you can use the haarcascades file I have attached to this folder.























Code For Facial Recognition

You can perform this by typing in the following code 


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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
           y1,x2,y2,x1 = faceLoc
           y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
           cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
           markAttendance(name)
  
   if cv2.waitKey(1) & 0XFF == ord ('o') :
       break
   cv2.imshow('Webcam', img)
   cv2.waitKey(1)

Note: Ensure the face capture storage folder and the facial recognition folder are the same.Do remember to create an attendance.csv file to store the recognition data.

Upon running this program you will be able to perform facial recognition as long as you have stored an image of yourself in there. Do remember to save the image name as your name as it identifies by using the image name as your name.

You can now perform facial recognition.



Fall Detection

8.1 Training dataset for fall Detection

To train a dataset for fall detection we can use that video guide again but this time using fall detection dataset that I have found online. https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset

You can download and use these files to train your dataset by following the video guide.

8.2 Creating the program

import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
   return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
   parser.add_argument('--camera', type=str, default=0)

   parser.add_argument('--resize', type=str, default='0x0',
                       help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
   parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                       help='if provided, resize heatmaps before they are post-processed. default=1.0')

   parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
   parser.add_argument('--show-process', type=bool, default=False,
                       help='for debug purpose, if enabled, speed for inference is dropped.')
  
   parser.add_argument('--tensorrt', type=str, default="False",
                       help='for tensorrt process.')
   parser.add_argument('--save_video',type= bool,default=False,
                       help='To write output video. default name file_name_output.avi')
   args = parser.parse_args()

   mode = int(input("Enter a mode :"))

   logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
   w, h = model_wh(args.resize)
   if w > 0 and h > 0:
       e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
   else:
       e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
   logger.debug('cam read+')
   cam = cv2.VideoCapture(args.camera)
   ret_val, image = cam.read()
   logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
   count = 0
   y1 = [0,0]
   frame = 0
   while True:
       ret_val, image = cam.read()
       i = 1
       logger.debug('image process+')
       humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

       logger.debug('postprocess+')
       image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

       if mode == 1:
           hu = len (humans)
           print("Total no. of people :", hu)
       elif mode == 2:
           for human in humans :
               for i in range(len(humans)):
                   try:
                       a = human.body_parts[0] #Head
                       x = a.x*image.shape[1]
                       y = a.y*image.shape[0]
                       y1.append(y)                  
                   except:
                       pass
                   if ((y - y1[-2]) > 30):
                       cv2.putText(image,"Fall Detected",(20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5,(255,255,0), 2, 11)
                       print("fall detected.", i+1)

       # logger.debug('show+')
       # no_people = len(humans)
       # print("number of people :",no_people)
       # cv2.putText(image,
       #             "Number of People: %d" % (no_people),
       #             (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
       #             (255, 255, 255), 2)
       cv2.putText(image,
                   "FPS: %f" % (1.0 / (time.time() - fps_time)),
                   (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0), 2)
       cv2.imshow('tf-pose-estimation result', image)
       fps_time = time.time()
       if(frame == 0) and(args.save_video):
           out = cv2.VideoWriter(file_write_name+'_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                   20,(image.shape[1],image.shape[0]))
           out.write(image)
       if cv2.waitKey(1) == 27:
           break
       logger.debug('finished+')

   cv2.destroyAllWindows()

You can now perform facial recognition upon running this program. If any of the import packages are missing or if the terminal says a dependency is missing, google that particular dependency or search it on pypi to install it. 


How does fall detection work?



The idea is fairly simple. There are 14 points all over the persons body and they are linked to his joints. Once the position of the point for the head is noticed to be lower than the other points. It is labelled as a fall.







Problems Encountered

One of the biggest problems that I have faced and you will too is the lack of resources. If you get stuck at any step finding your way out is going to be quite a challenge. You might even have to go as far as contacting Nvidia support themselves for help. Do not give up as the process is difficult but worthwhile.

Another one of the issues you might face is installing the wrong library version, python version and dependencies. So far python 3.9 has been working with this project but in the future this might not be the case.

Lastly you will have to go through a lot of errors initially as there is a high chance that something might not work, googling has proven to be really helpful as most of the solutions have been discovered either on stackoverflow or other forums.







This ia timeline of how long I took me to create this project from scratch. For the user of this guide it should not take more than a few weeks as most of the research has been done however incase of any issues you can approach me on my github or take a look at the completed project that I have uploaded to my github.

https://github.com/meethu51











Conclusion

I am very grateful to have this opportunity to be able to work with such great mentors and such high end devices to complete such an impressive project. This project has helped me secure many internships at famous companies. All I have to say that without my benefactors I would not have been able to come this far and complete such a strenuous project all by myself from scratch.
