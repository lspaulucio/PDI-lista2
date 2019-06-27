# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""

import numpy as np
from PIL import Image
from copy import deepcopy
import MyLib as ml

img = Image.open('images/peppers.tiff')             # Le a imagem
img = np.array(img)                                 # Converte a imagem para um numpy array
orig = deepcopy(img)
chR = img[:, :, 0]
chG = img[:, :, 1]
chB = img[:, :, 2]

THRESHOLD = 20
index = chR > (chG + THRESHOLD)    # Finding regions where red color is bigger than green color by 20
index2 = chR > (chB + THRESHOLD)   # Finding regions where red color is bigger than blue color by 20

x = index & index2
chR[x], chB[x] = chB[x], chR[x]  # Changing masks

g = [orig, img]
title = ['Imagem Original', 'Resultado Obtido']
ml.show_images(g, 1, title)
