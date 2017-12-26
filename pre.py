"""
   pre

   Copyright 2017 Weverson Nascimento

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__author__ = "Weverson Nascimento"
__credits__ = ["Weverson Nascimento"]
__license__ = "apache-2.0"
__version__ = "0.1"
__maintainer__ = "Weverson Nascimento"
__email__ = "weverson@ufpa.br"
__status__ = "Production"

'''
    File name: pre.py
    Author: Weverson Nascimento
    Date created: 24/09/2017
    Date last modified: 25/12/2017
    Python Version: 2.7
'''

import numpy as np
import cv2


def pre(caminho):
    # importa os classificadores
    face_cascade = cv2.CascadeClassifier('/home/weverson/anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml') # arquivo do classificador de face
    eye_cascade = cv2.CascadeClassifier('/home/weverson/anaconda2/share/OpenCV/haarcascades/haarcascade_eye.xml') # arquivo do classificador de olhos

    # le a imagem de entrada
    img = cv2.imread(caminho)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte para tons de cinza
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # detecta faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # draw a rectangle on  each face detected
        roi_gray = gray[y:y+h, x:x+w] # crop gray image
        roi_color = img[y:y+h, x:x+w] # crop original image
        resized_img = cv2.resize(roi_gray, (67,55))
        eyes = eye_cascade.detectMultiScale(roi_gray) # detect eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # draw a rectangle on each eye detected

    return resized_img
