# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image
import MyLib as ml
import matplotlib.pyplot as plt

ABSOLUTE_ZERO = np.array([[0, 72, 186]])
AMARANTH_PINK = np.array([[241, 156, 187]])
BLUE = np.array([[0, 0, 255]])
ACID_GREEN = np.array([[176, 191, 26]])
GREEN = np.array([[0, 255, 0]])
ALLOY_ORANGE = np.array([[196, 98, 16]])
AMBER = np.array([[255, 191, 0]])
RED = np.array([[255, 0, 0]])
WHITE = np.array([[255, 255, 255]])
BLACK = np.array([[0, 0, 0]])

img = Image.open('images/Thyroid.jpg')
img = img.convert('L')
img = np.array(img)
histograma = ml.calculaHistograma(img)

imgRGB = Image.fromarray(img).convert('RGB')
imgRGB = np.array(imgRGB)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] < 25:
            imgRGB[i, j] = BLACK
        elif img[i, j] < 40:
            imgRGB[i, j] = ABSOLUTE_ZERO
        elif img[i, j] < 60:
            imgRGB[i, j] = AMARANTH_PINK
        elif img[i, j] < 80:
            imgRGB[i, j] = ACID_GREEN
        elif img[i, j] < 100:
            imgRGB[i, j] = ALLOY_ORANGE
        elif img[i, j] < 120:
            imgRGB[i, j] = AMBER
        elif img[i, j] < 140:
            imgRGB[i, j] = RED
        else:
            imgRGB[i, j] = WHITE

g = [img, imgRGB]
ml.show_images(g)
