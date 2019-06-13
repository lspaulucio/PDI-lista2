# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image
import MyLib as ml
from copy import deepcopy

E3 = np.ones((3, 3))
E7 = np.ones((7, 7))


def erosao(img, elem):

    pad = elem.shape[0]
    paddedImg = ml.paddingImage(img, padding_size=pad // 2)
    imgNew = np.zeros(img.shape)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            non_zeros = np.count_nonzero(elem * paddedImg[i:i+pad, j:j+pad])
            if non_zeros == (pad*pad):
                imgNew[i, j] = 255
            else:
                imgNew[i, j] = 0

    return imgNew


def dilatacao(img, elem):

    pad = elem.shape[0]
    paddedImg = ml.paddingImage(img, padding_size=(pad//2))
    imgNew = np.zeros(img.shape)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            non_zeros = np.count_nonzero(elem * paddedImg[i:i+pad, j:j+pad])
            if non_zeros == 0:
                imgNew[i, j] = 0
            else:
                imgNew[i, j] = 255

    return imgNew


def abertura(img, elem):
    newImg = erosao(img, elem)
    newImg = dilatacao(newImg, elem)
    return newImg


def fechamento(img, elem):
    newImg = dilatacao(img, elem)
    newImg = erosao(newImg, elem)
    return newImg


def esqueleto(img, elem):

    imgCopy = deepcopy(img)
    s = []
    i, k = 0, 0
    s.append(imgCopy)

    # OLHAR LOOP
    while(True):
        imgCopy = erosao(imgCopy, elem)
        if np.count_nonzero(imgCopy) == 0:
            k = i - 1
            print("Kmax: {}".format(k))
            break
        s.append(imgCopy)
        i += 1

    sk = np.zeros(img.shape)
    for i in s:
        sk += i - abertura(i, elem)
        sk[sk > 255] = 255
        sk[sk < 0] = 0

    return sk


img = Image.open('images/Fig11.10.jpg')
img = np.array(img)

skeleton = esqueleto(img, E3)

r = img - skeleton
g = [img, skeleton, r]
ml.show_images(g)


# imgBorder = ml.conv2D(img, ml.LAPLACE_FILTER)
#
# borders = [(i, j) for i in range(0, imgBorder.shape[0]) for j in range(0, imgBorder.shape[1]) if imgBorder[i, j] == 255]
#
# windows = [getWindow(img, i, j) for i, j in borders]
# N = [np.count_nonzero(window) for window in windows]
# N = np.array(N)
# idx = N <= 6
# idx1 = N >= 2
# idx = idx & idx1
# G = N[idx]
# print(N)
# Image.fromarray(img).show()
