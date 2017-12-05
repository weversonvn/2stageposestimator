__author__ = "Weverson Nascimento"
__credits__ = ["Weverson Nascimento"]
__license__ = "apache-2.0"
__version__ = "1.0.1"
__maintainer__ = "Weverson Nascimento"
__email__ = "weverson@ufpa.br"
__status__ = "Production"

'''
    File name: teste.py
    Author: Weverson Nascimento
    Date created: 24/09/2017
    Date last modified: 25/11/2017
    Python Version: 2.7
'''


'''
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
'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
import bob.ip.gabor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')


img = cv2.imread('/home/weverson/Downloads/HeadPoseImageDatabase/Front/personne01146+0+0.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
gwt = bob.ip.gabor.Transform(number_of_scales = 6)
trafo_image = gwt(gray)


# plot the results of the transform for some wavelets
for scale in (0,2,4):
  for direction in (0,2,4):
    plt.subplot(3,6,4+scale*3+direction/2)
    plt.imshow(np.real(trafo_image[scale*gwt.number_of_directions+direction]), cmap='gray')
    plt.title("Scale %d, direction %d" % (scale, direction))
    plt.gca().invert_yaxis()

plt.show()
