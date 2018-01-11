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
    Date last modified: 11/01/2018
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


def treino(mat_paths, p): # do the whole thing
    prototipo = np.empty([48,93,p])
    kpca = np.empty([48],dtype=object)
    mat_magnitudes = np.empty([48,93,67*55])
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        magnitudes = np.empty([48,15,67*55])
        for person in range(15):
            img_cropped = pre(mat_paths[pose,person])
            trafo_image = wavextract(img_cropped)
            for rotation in range(48):
                magnitudes[rotation,person] = np.reshape(np.abs(trafo_image[rotation]),67*55)
        for rotation in range(48):
            mat_magnitudes[rotation,pose] = np.mean(magnitudes[rotation,:],axis=0)
            
    print 'Faz a projecao'
    for rotation in range(48):
        print 'Rotacao ' + str(rotation) + '/47'
        wav_coefs = mat_magnitudes[rotation]
        prototipo[rotation], kpca[rotation] = projecaokpca(wav_coefs,p)
    
    print 'Salvando...'
    with open('treino.pkl','w') as f:     # salva no arquivo treino.pkl
        pickle.dump([prototipo, kpca], f) # as variaveis prototipo e kpca


def teste(caminho, prototipo, kpca):
    mat_paths = readpath(caminho, 1)
    dk = np.empty([93*15,48],dtype=int)
    mat_magnitudes = np.empty([48,93*15,67*55])
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            img_cropped = pre(mat_paths[pose,person])
            trafo_image = wavextract(img_cropped)
            for rotation in range(48):
                magnitude = np.reshape(np.abs(trafo_image[rotation]),67*55)
                mat_magnitudes[rotation,15*pose+person] = magnitude
    
    print 'Calcula dk'
    y = np.empty([48,93*15,1])
    for rotation in range(48):
        y[rotation] = kpca[rotation].transform(mat_magnitudes[rotation])
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            for rotation in range(48):
                img = 15*pose+person
                d = np.abs(y[rotation,img]-prototipo[rotation])
                dk[img,rotation] = np.argmin(d)
    
    print 'Calcula acertos'
    acerto = np.empty([93*15],dtype=bool)
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            c = np.argmax(np.bincount(dk[15*pose+person])) # anota a classe
            acerto[15*pose+person] = c==pose
    acertos = np.count_nonzero(acerto)
    taxa = (acertos*100)/(93*15);
    print 'Acertos = ' + str(acertos) + ' de 1395 (' + str(taxa) + '%)'


if __name__ == '__main__':
    try:
        caminho = sys.argv[1] # o caminho do dataset de imagens
    except:
        caminho = raw_input("Insira o caminho do dataset de imagens: ")
    p = 1 # numero de autovetores da projecao kpca
    try:
        with open('treino.pkl') as f:
            prototipo, kpca = pickle.load(f)
    except IOError:
        print "Arquivo de treino nao encontrado. Treinando..."
        mat_paths = readpath(caminho, 0) # 0 para treino, 1 para teste
        treino(mat_paths, p)
        with open('treino.pkl') as f:
            prototipo, kpca = pickle.load(f)
    print 'Testando...'
    teste(caminho, prototipo, kpca)

