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
    Date last modified: 17/01/2018
    Python Version: 2.7
'''


import sys
import numpy as np
import pickle # para salvar em arquivo as variaveis

import cv2
import matplotlib.pyplot as plt
import bob.ip.gabor
from sklearn.decomposition import KernelPCA

from pre import preds
from kpcasub import projecaokpca

#print(__doc__)


def wavextract(imagem):
    # extracao de coeficientes com wavelet de Gabor
    escalas = 6
    gwt = bob.ip.gabor.Transform(number_of_scales = escalas) # cria a transformada
    trafo_image = gwt(imagem) # aplica a transformada na imagem
    return trafo_image


def readpath(caminho, n): # adaptado de Gourier2004 (disponivel com o dataset)
    pessoa = 1
    serie = 1
    mat_paths = np.empty([93,15],dtype=object)
    mat_pre = np.empty([93,15,4],dtype=int)
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
                arquivo = caminho + 'Person' + str(pessoa).zfill(2) + '/person' + str(pessoa).zfill(2) + str(serie) + str(i).zfill(2) + tiltplus + str(tilt) + panplus + str(pan)
                mat_paths[i,pessoa-1] = arquivo + '.jpg'
                txt = arquivo + '.txt'
                f = open(txt,'r')
                k = 0
                for line in f:
                    k = k + 1
                    if k == 4:
                        mat_pre[i,pessoa-1,0] = int(line)
                    elif k == 5:
                        mat_pre[i,pessoa-1,1] = int(line)
                    elif k == 6:
                        mat_pre[i,pessoa-1,2] = int(line)
                    elif k == 7:
                        mat_pre[i,pessoa-1,3] = int(line)
    return mat_paths, mat_pre


def treino(caminho, p): # treina o classificador
    mat_paths, mat_pre = readpath(caminho, 0) # 0: treino, 1: teste
    prototipo = np.empty([48,93,p]) # o prototipo que sera comparado com a imagem de teste
    kpca = np.empty([48],dtype=object) # o objeto que aplica a transformacao kpca no teste
    mat_magnitudes = np.empty([48,93*15,67*55]) # as magnitudes de transformada de gabor
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            img_cropped = preds(mat_paths[pose,person], mat_pre[pose,person]) # a imagem da face
            trafo_image = wavextract(img_cropped) # a transf. de Gabor
            for rotation in range(48):
                magnitude = np.reshape(np.abs(trafo_image[rotation]),67*55) # vetoriza a imagem
                mat_magnitudes[rotation,15*pose+person] = magnitude # salva a magnitude na matriz
    del mat_paths, img_cropped, trafo_image, magnitude
            
    print 'Faz a projecao'
    for rotation in range(48):
        print 'Rotacao ' + str(rotation) + '/47'
        wav_coefs = mat_magnitudes[rotation] # coeficientes por rotacao
        projecoes, kpca[rotation] = projecaokpca(wav_coefs,p) # treina o classificador
        for pose in range(93):
            prototipo[rotation,pose] = np.mean(projecoes[15*pose:15*pose+15],axis=0)
    del mat_magnitudes, wav_coefs, projecoes
    
    print 'Salvando...'
    with open('treino.pkl','w') as f:     # salva no arquivo treino.pkl
        pickle.dump([prototipo, kpca], f) # as variaveis prototipo e kpca
    print 'Testando...'
    teste(caminho, prototipo, kpca)


def teste(caminho, prototipo, kpca): # testa o classificador
    mat_paths, mat_pre = readpath(caminho, 1)
    mat_magnitudes = np.empty([48,93*15,67*55])
    dk = np.empty([93*15,48],dtype=int) # armazena as poses estimadas por rotacao para cada imagem
    
    print 'Calcula magnitudes das imagens'
    for pose in range(93):
        print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            img_cropped = preds(mat_paths[pose,person], mat_pre[pose,person])
            trafo_image = wavextract(img_cropped)
            for rotation in range(48):
                magnitude = np.reshape(np.abs(trafo_image[rotation]),67*55)
                mat_magnitudes[rotation,15*pose+person] = magnitude
    del mat_paths, img_cropped, trafo_image, magnitude
    
    print 'Determina classes esperadas'
    y = np.empty([48,93*15,p]) # os prototipos de teste por resolucao
    for rotation in range(48):
        print 'Rotacao ' + str(rotation) + '/47'
        y[rotation] = kpca[rotation].transform(mat_magnitudes[rotation]) # aplica o kpca na resolucao correspondente
    for pose in range(93):
        for person in range(15):
            for rotation in range(48):
                img = 15*pose+person
                d = np.linalg.norm(y[rotation,img]-prototipo[rotation],axis=1) # calcula a norma da diferenca de y com cada prototipo
                dk[img,rotation] = np.argmin(d) # a pose estimada tem a menor norma
    
    print 'Calcula acertos'
    acerto = np.empty([93*15],dtype=bool)
    for pose in range(93):
        for person in range(15):
            img = 15*pose+person
            c = np.argmax(np.bincount(dk[img])) # anota a classe prevista
            acerto[img] = c==pose # compara a classe prevista com a real
    acertos = np.count_nonzero(acerto) # conta a quantidade de acertos
    taxa = (acertos*100)/(93*15); # calcula a taxa de acertos
    print 'Acertos = ' + str(acertos) + '/1395 (' + str(taxa) + '%)'


if __name__ == '__main__':
    try:
        caminho = sys.argv[1] # o caminho do dataset de imagens
    except:
        caminho = raw_input("Insira o caminho do dataset de imagens: ")
    p = 10 # numero de autovetores da projecao kpca
    print 'Verificando presenca do arquivo de treino'
    try:
        with open('treino.pkl') as f:
            prototipo, kpca = pickle.load(f)
        print 'Testando...'
        teste(caminho, prototipo, kpca)
    except IOError:
        print "Arquivo de treino nao encontrado. Treinando..."
        treino(caminho, p)
            

