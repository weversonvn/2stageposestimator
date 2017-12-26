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
    Date last modified: 25/12/2017
    Python Version: 2.7
'''


import sys
import numpy as np

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
    rotacoes = gwt.number_of_directions
    trafo_image = gwt(imagem) # aplica a transformada na imagem
    for escala in range(escalas):
        for rotacao in range(rotacoes):
            real = np.real(trafo_image[escala*rotacoes+rotacao]) # extrai parte real
            wav_coefs = np.reshape(real,3685) # vetoriza a matriz
    return wav_coefs


def readimg(caminho):
    pessoa = 1
    serie = 1
    wav_coefs = np.empty([0,3685])
    for pessoa in range(1,16):
        for serie in range(1,3):
            for i in range(93):
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
                imgfile = caminho + 'Person' + str(pessoa).zfill(2) + '/person' + str(pessoa).zfill(2) + str(serie) + str(i).zfill(2) + tiltplus + str(tilt) + panplus + str(pan) + '.jpg'
                imagem = pre(imgfile)
                wav_coefs = np.append(wav_coefs,[wavextract(imagem)],axis=0)
    return wav_coefs

    
if __name__ == '__main__':
    caminho = sys.argv[1] # o caminho do dataset de imagens
    wav_coefs = readimg(caminho)
    print wav_coefs.shape

