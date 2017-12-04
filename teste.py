__author__ = "Weverson Nascimento"
__credits__ = ["Weverson Nascimento"]
__license__ = "GPL"
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
