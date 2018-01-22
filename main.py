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
    Date last modified: 22/01/2018
    Python Version: 2.7
'''


import sys
import numpy as np
import pickle # para salvar em arquivo as variaveis

import cv2
import matplotlib.pyplot as plt
import bob.ip.gabor

from pre import preds
from classifier import projecaokpca, projecaosvc, projecaolda

#print(__doc__)


def wavextract(imagem):
    # extracao de coeficientes com wavelet de Gabor
    escalas = 6
    gwt = bob.ip.gabor.Transform(number_of_scales = escalas) # cria a transformada
    trafo_image = gwt(imagem) # aplica a transformada na imagem
    magnitude = np.reshape(np.abs(trafo_image),(48,67*55))
    return magnitude


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


def magnitudes(mat_paths, mat_pre, poses, classes):
    mat_magnitudes = np.empty([48,classes*15,67*55]) # as magnitudes de transformada de gabor
    y = np.empty([classes*15],dtype=int) # a classe correspondente das imagens
    cont = 0
    print 'Calcula magnitudes das imagens'
    for pose in poses:
        #print 'Pose ' + str(pose) + '/92'
        for person in range(15):
            index = 15*cont+person
            y[index] = pose
            img = mat_paths[pose,person]
            img_cropped = preds(img, mat_pre[pose,person]) # a imagem da face
            magnitude = wavextract(img_cropped) # a transf. de Gabor
            for rotation in range(48):
                mat_magnitudes[rotation,index] = magnitude[rotation] # salva a magnitude na matriz
        cont += 1
    return mat_magnitudes, y

def calculacertos(dk, classificador, poses):
    classes = len(poses)
    acerto = np.empty([classes*15],dtype=bool)
    print 'Calcula acertos'
    if classificador == 'kpca':
        for pose in range(classes):
            for person in range(15):
                img = 15*pose+person
                c = np.argmax(np.bincount(dk[img])) # anota a classe prevista
                acerto[img] = c==pose # compara a classe prevista com a real
    else:
        cont = 0
        for pose in poses:
            for person in range(15):
                img = 15*cont+person
                c = np.argmax(np.bincount(dk[img])) # anota a classe prevista
                acerto[img] = c==pose # compara a classe prevista com a real
            cont += 1
    acertos = np.count_nonzero(acerto) # conta a quantidade de acertos
    taxa = (acertos*100)/(classes*15); # calcula a taxa de acertos
    return acertos, taxa


def treino(caminho, classificador, conj=None, p=10): # treina o classificador
    if not conj == None:
        poses = conj
    else:
        poses = range(93)
    classes = len(poses)
    mat_paths, mat_pre = readpath(caminho, 0) # 0: treino, 1: teste
    prototipo = np.empty([48,classes,p]) # o prototipo que sera comparado com a imagem de teste
    clf = np.empty([48],dtype=object) # o objeto que aplica a transformacao kpca no teste
    mat_magnitudes, y = magnitudes(mat_paths, mat_pre, poses, classes) # obtem as magnitudes da imagem
    
    print 'Treina o classificador ' + classificador
    if classificador == 'kpca':
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            wav_coefs = mat_magnitudes[rotation] # coeficientes por rotacao
            projecoes, clf[rotation] = projecaokpca(wav_coefs,p) # treina o classificador
            cont = 0
            for pose in poses:
                interval = projecoes[15*cont:15*cont+15]
                prototipo[rotation,cont] = np.mean(interval,axis=0)
                cont += 1
        del mat_magnitudes, wav_coefs, y, projecoes
    
        print 'Salvando...'
        with open('treinokpca.pkl','w') as f:     # salva arquivo de treino
            pickle.dump([prototipo, clf], f)
        
        print 'Testando...'
        teste(caminho, classificador, clf, conj, prototipo)
    
    elif classificador == 'svc':
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            wav_coefs = mat_magnitudes[rotation] # coeficientes por rotacao
            clf[rotation] = projecaosvc(wav_coefs,y) # treina o classificador
        del mat_magnitudes, wav_coefs, y
        
        print 'Salvando...'
        with open('treinosvc.pkl','w') as f:     # salva arquivo de treino
            pickle.dump(clf, f)
        
        print 'Testando...'
        teste(caminho, classificador, clf, conj)
    
    elif classificador == 'lda':
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            wav_coefs = mat_magnitudes[rotation] # coeficientes por rotacao
            clf[rotation] = projecaolda(wav_coefs,y) # treina o classificador
        del mat_magnitudes, wav_coefs, y
        
        print 'Salvando...'
        with open('treinolda.pkl','w') as f:     # salva arquivo de treino
            pickle.dump(clf, f)
        
        print 'Testando...'
        teste(caminho, classificador, clf, conj)


def teste(caminho, classificador, clf, conj=None, prototipo=None): # testa o classificador
    if not conj == None:
        poses = conj
    else:
        poses = range(93)
    classes = len(poses)
    mat_paths, mat_pre = readpath(caminho, 1)
    mat_magnitudes, y = magnitudes(mat_paths, mat_pre, poses, classes) # obtem as magnitudes da imagem
    
    print 'Determina classes esperadas ' + classificador
    if classificador == 'kpca':
        dk = np.empty([classes*15,48],dtype=int) # armazena as poses estimadas por rotacao para cada imagem
        y = np.empty([48,classes*15,p]) # os prototipos de teste por resolucao
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            y[rotation] = clf[rotation].transform(mat_magnitudes[rotation]) # aplica o kpca na resolucao correspondente
        cont = 0
        for pose in poses:
            for person in range(15):
                for rotation in range(48):
                    img = 15*cont+person
                    d = np.linalg.norm(y[rotation,img]-prototipo[rotation],axis=1) # calcula a norma da diferenca de y com cada prototipo
                    dk[img,rotation] = np.argmin(d) # a pose estimada tem a menor norma
            cont += 1
    
    elif classificador == 'svc':
        y = np.empty([48,classes*15],dtype=int) # a classe esperada pelo classificador
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            y[rotation] = clf[rotation].predict(mat_magnitudes[rotation])
        dk = y.T # transpoe a matriz para analizar por resolucao e nao por img
    
    elif classificador == 'lda':
        y = np.empty([48,classes*15],dtype=int) # a classe esperada pelo classificador
        for rotation in range(48):
            #print 'Rotacao ' + str(rotation) + '/47'
            y[rotation] = clf[rotation].predict(mat_magnitudes[rotation])
        dk = y.T # transpoe a matriz para analizar por resolucao e nao por img
    
    acertos, taxa = calculacertos(dk, classificador, poses)
    taxas[indice] = taxa
    print 'Acertos ' + classificador + ' = ' + str(acertos) + '/' + str(classes*15) + ' (' + str(taxa) + '%)'


if __name__ == '__main__':
    try:
        caminho = sys.argv[1] # o caminho do dataset de imagens
    except:
        caminho = raw_input("Insira o caminho do dataset de imagens: ")
    p = 10 # numero de autovetores da projecao kpca
    conj = [15,45,77,82] # classes usadas no treino e teste
    print "Quantidade de classes: " + str(len(conj))
    indice = 0 # indice do vetor de taxas
    taxas = np.empty(3) # armazena as taxas de acerto
    for classificador in ('kpca', 'svc', 'lda'):
        trainfile = 'treino' + classificador + '.pkl'
        prototipo = None
        try:
            with open(trainfile) as f:
                print 'Carregando arquivo de treino'
                if classificador == 'kpca':
                    prototipo, clf = pickle.load(f)
                else:
                    clf = pickle.load(f)
            print 'Testando...'
            teste(caminho, classificador, clf, conj, prototipo)
        except IOError:
            print "Arquivo de treino nao encontrado. Treinando..."
            treino(caminho, classificador, conj, p)
        indice += 1
    fig, ax = plt.subplots() # cria figura para plot de acertos
    plt.bar(np.arange(1,4),taxas)
    ax.set_xticks(np.arange(1,4))
    ax.set_xticklabels(['KPCA','SVC','LDA'])
    ax.set_ylabel('Porcentagem de acertos')
    ax.set_title('Acertos com quantidade de classes = ' + str(len(conj)))
    filename = 'fig' + str(len(conj)-1) + '.pdf'
    fig.savefig(filename)
    plt.show()
