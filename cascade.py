import numpy as np
import cv2
import matplotlib.pyplot as plt

# import classifiers
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

# import image
img = cv2.imread('/home/weverson/Downloads/HeadPoseImageDatabase/Person01/person01120-30+0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # detect faces
for (x,y,w,h) in faces:
     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # draw a rectangle on  each face detected
     roi_gray = gray[y:y+h, x:x+w] # crop gray image
     roi_color = img[y:y+h, x:x+w] # crop original image
     eyes = eye_cascade.detectMultiScale(roi_gray) # detect eyes
     for (ex,ey,ew,eh) in eyes:
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # draw a rectangle on each eye detected

print np.shape(roi_color)

cv2.imshow('Imagem',img) # show image
cv2.waitKey(0) # wait for a key to exit
cv2.destroyAllWindows() # exit
