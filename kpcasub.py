"""
   kpcasub

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
    File name: kpcasub.py
    Author: Weverson Nascimento
    Date created: 14/12/2017
    Date last modified: 20/12/2017
    Python Version: 2.7
'''


from sklearn.decomposition import KernelPCA

def projecaokpca(X):
    kpca = KernelPCA(n_components=30, kernel="rbf", n_jobs=-1)
    coeficientes = np.empty([escalas*rotacoes,55,55])
    for escala in range(escalas):
        for rotacao in range(rotacoes):
            # para cada rotacao, cria prototipos
            real = np.real(X[escala*rotacoes+rotacao])
            coeficientes[escala*rotacoes+rotacao]= kpca.fit_transform(real)

    return coeficientes