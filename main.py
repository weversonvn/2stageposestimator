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
    Date last modified: 14/01/2018
    Python Version: 2.7
'''


import sys
import numpy as np
import pickle # para salvar em arquivo as variaveis

import cv2
import matplotlib.pyplot as plt
import bob.ip.gabor
from sklearn.svm import SVC

from pre import pre
from kpcasub import projecaokpca

#print(__doc__)


def wavextract(imagem):
    # extracao de coeficientes com wavelet de Gabor
    escalas = 6
    gwt = bob.ip.gabor.Transform(number_of_scales = escalas) # cria a transformada
    trafo_image = gwt(imagem) # aplica a transformada na imagem
    return trafo_image


def projecaosvc(wav_coefs, y):
    clf = SVC()
    classificador = clf.fit(wav_coefs,y)
    return classificador


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


def treino(mat_paths): # treina o classificador
    clf = np.empty([48],dtype=object) # o classificador svc por resolucao
    mat_magnitudes = np.empty([48,93,67*55]) # as magnitudes de transformada
    y = np.empty([93*15],dtype=int) # a classe correspondente das imagens
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        magnitudes = np.empty([48,15,67*55])
        for person in range(15):
            img_cropped = pre(mat_paths[pose,person]) # a imagem da face
            trafo_image = wavextract(img_cropped) # a transf. de Gabor
            y[15*pose+person] = pose # a classe dessa imagem
            for rotation in range(48):
                magnitudes[rotation,person] = np.reshape(np.abs(trafo_image[rotation]),67*55) # vetoriza a imagem
            
    print 'Faz a projecao'
    for rotation in range(48):
        print 'Rotacao ' + str(rotation) + '/47'
        wav_coefs = mat_magnitudes[rotation] # coeficientes por rotacao
        clf[rotation] = projecaosvc(wav_coefs,y) # treina o classificador
    
    print 'Salvando...'
    with open('treino.pkl','w') as f: # salva no arquivo treino.pkl
        pickle.dump(clf, f)           # a variavel clf


def teste(caminho, clf):
    mat_paths = readpath(caminho, 1)
    mat_magnitudes = np.empty([48,93*15,67*55])
    y = np.empty([48,93*15],dtype=int) # a classe esperada pelo classificador
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            img_cropped = pre(mat_paths[pose,person])
            trafo_image = wavextract(img_cropped)
            for rotation in range(48):
                magnitude = np.reshape(np.abs(trafo_image[rotation]),67*55)
                mat_magnitudes[rotation,15*pose+person] = magnitude
    
    print 'Determina classes esperadas'
    for rotation in range(48):
        print 'Rotacao ' + str(rotation) + '/47'
        y[rotation] = clf[rotation].predict(mat_magnitudes[rotation])
    d = y.T # transpoe a matriz para analizar por resolucao e n por img
    print 'Calcula acertos'
    acerto = np.empty([93*15],dtype=bool)
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            img = 15*pose+person
            c = np.argmax(np.bincount(d[img])) # anota a classe prevista
            acerto[1img] = c==pose # compara a classe prevista com a real
    acertos = np.count_nonzero(acerto) # conta a quantidade de acertos
    taxa = (acertos*100)/(93*15);
    print 'Acertos = ' + str(acertos) + '/1395 (' + str(taxa) + '%)'


if __name__ == '__main__':
    try:
        caminho = sys.argv[1] # o caminho do dataset de imagens
    except:
        caminho = raw_input("Insira o caminho do dataset de imagens: ")
    p = 1 # numero de autovetores da projecao kpca
    try:
        with open('treino.pkl') as f:
            clf = pickle.load(f)
    except IOError:
        print "Arquivo de treino nao encontrado. Treinando..."
        mat_paths = readpath(caminho, 0) # 0 para treino, 1 para teste
        treino(mat_paths, p)
        with open('treino.pkl') as f:
            clf = pickle.load(f)
    print 'Testando...'
    teste(caminho, clf)

