"""
   2stageposestimator

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
    File name: main.py
    Author: Weverson Nascimento
    Date created: 24/09/2017
    Date last modified: 03/01/2018
    Python Version: 2.7
'''


import sys
import numpy as np
import pickle # para salvar em arquivo as variaveis

import cv2
import matplotlib.pyplot as plt
import bob.ip.gabor
from sklearn.decomposition import KernelPCA

from pre import pre
from kpcasub import projecaokpca

#print(__doc__)


def wavextract(imagem):
    # extracao de coeficientes com wavelet de Gabor
    escalas = 6
    gwt = bob.ip.gabor.Transform(number_of_scales = escalas) # cria a transformada
    trafo_image = gwt(imagem) # aplica a transformada na imagem
    #return trafo_img, escalas, gwt.number_of_directions
    return trafo_image


def readpath(caminho, n):
    pessoa = 1
    serie = 1
    mat_paths = np.empty([93,15],dtype=object)
    for i in range(93):
        for serie in range(n+1,n+2):
            for pessoa in range(1,16):
                panplus = ""
                tiltplus = ""
                if i == 0:
                    tilt = -90
                    pan = 0
                elif i == 92:
                    tilt = 90
                    pan = 0
                else :
                    pan = ((i-1) % 13-6)*15
                    tilt = ((i-1) / 13-3)*15
                    if (abs(tilt) == 45):
                        tilt = tilt / abs(tilt)*60
                if pan >= 0:
                    panplus = "+"
                if tilt >= 0:
                    tiltplus = "+"
                mat_paths[i,pessoa-1] = caminho + 'Person' + str(pessoa).zfill(2) + '/person' + str(pessoa).zfill(2) + str(serie) + str(i).zfill(2) + tiltplus + str(tilt) + panplus + str(pan) + '.jpg'
    return mat_paths


def dtwt(mat_paths): # do the whole thing
    prototipo = np.empty([10,93,1])
    kpca = np.empty([10],dtype=object)
    for rotation in range(10):
        wav_coefs = np.empty([0])
        for pose in range(93):
            print 'rotacao ' + str(rotation) + '/9 pose ' + str(pose) + '/92'
            wav_mean = np.empty([0])
            for person in range(15):
                img_cropped = pre(mat_paths[pose,person])
                trafo_image = wavextract(img_cropped)
                magnitude = np.linalg.norm(trafo_image[rotation]) # usa somente a magnitude
                #wav_vet = np.reshape(magnitude,3685) # vetoriza a matriz
                wav_mean = np.append(wav_mean,[magnitude],axis=0)
            wav_mean_vet = np.mean(wav_mean,axis=0)
            wav_coefs = np.append(wav_coefs,[wav_mean_vet],axis=0)
        wav_coefs = np.reshape(wav_coefs,(93,1))
        prototipo[rotation], kpca[rotation] = projecaokpca(wav_coefs)
    with open('treino.pkl','w') as f:     # salva no arquivo treino.pkl
        pickle.dump([prototipo, kpca], f) # as variaveis prototipo e kpca


if __name__ == '__main__':
    caminho = sys.argv[1] # o caminho do dataset de imagens
    mat_paths = readpath(caminho, 0) # 0 para treino, 1 para teste
    dtwt(mat_paths)

