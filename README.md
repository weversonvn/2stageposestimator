# 2stageposestimator

### Descrição

Aplicação para estimação de posicionamento de face usando algoritmo de duas etapas.

### Configuração

São necessárias as seguintes bibliotecas:

* [Numpy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)
* [OpenCV](https://opencv.org/)
* [Bob's Gabor wavelet routines](https://pythonhosted.org/bob.ip.gabor/)
* [scikit-learn](http://scikit-learn.org/)

A biblioteca **Bob's Gabor wavelet routines** é mais facilmente instalada usando [Anaconda](https://www.anaconda.com/). Para instalação via Anaconda:

1. Adicione o repositório que contem a biblioteca: `conda config --add channels https://www.idiap.ch/software/bob/conda`;
2. Instale usando `conda install bob.ip.gabor`.

Se julgar necessário, crie um ambiente separado no Anaconda para executar todas as instalações necessárias.

##### Importante:

O OpenCV instalado com Anaconda não utiliza GTK, que é necessário para a criação das janelas de exibição de imagem. Por isso, é necessário instalar o OpenCV manualmente via código fonte habilitando o uso do GTK, conforme procedimento a seguir:

1. Instale as dependências do OpenCV conforme descrito na documentação;
2. Baixe o código fonte do site do OpenCV e extraia o mesmo;
3. Dentro da pasta extraída, crie um diretório para armazenar o código compilado `mkdir build` e entre na pasta;
4. Rode o comando `cmake` semelhante é explicado na documentação do OpenCV, incluindo como parâmetros:
- o caminho para a instalação do Anaconda (se você fez a instalação padrão do Anaconda, provavelmente será algo como `-D CMAKE_INSTALL_PREFIX=/home/usuario/anaconda2/`);
- o caminho do executável do Python (o executável do Anaconda, provavelmente será algo como `-D PYTHON2_EXECUTABLE=/home/usuario/anaconda2/bin/python`);
- o comando para usar GTK `WITH_GTK WITH_GTK_2_X`
5. O comando final seria algo como `cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/usuario/anaconda2/ -D PYTHON2_EXECUTABLE=/home/usuario/anaconda2/bin/python WITH_GTK WITH_GTK_2_X ..`.
6. Confirme, ao final da execução do `cmake`, se entre os itens que serão compilados está o python. Eventualmente, o python é detectado mas seus componentes não, então seria necessário incluir o restante dos parâmetros manualmente;
7. Compilar a biblioteca com `make -j2`;
8. Instalar com `sudo make install`;
9. Executar os comandos `echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf` e `sudo ldconfig` (o primeiro comando pode dar erro, caso ocorra ignore e execute o segundo comando normalmente).

### Uso:

Execute `python main.py caminho`, onde __caminho__ é o diretório com o dataset de imagens.

### Licença:

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
