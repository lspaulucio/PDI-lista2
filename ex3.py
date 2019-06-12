# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image
import MyLib as ml


def getWindow(img, i, j):
    step = 1
    window = img[i-step:i+step+1, j-step:j+step+1]
    window[1, 1] = 0
    return window


img = Image.open('images/Fig11.10.jpg')
img = np.array(img)

imgBorder = ml.conv2D(img, ml.LAPLACE_FILTER)

borders = [(i, j) for i in range(0, imgBorder.shape[0]) for j in range(0, imgBorder.shape[1]) if imgBorder[i, j] == 255]

windows = [getWindow(img, i, j) for i, j in borders]
N = [np.count_nonzero(window) for window in windows]
N = np.array(N)
idx = N <= 6
idx1 = N >= 2
idx = idx & idx1
G = N[idx]
print(N)
Image.fromarray(img).show()
