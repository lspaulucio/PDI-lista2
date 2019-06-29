# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 30/06/19
"""


import numpy as np
from PIL import Image
import MyLib as ml
from copy import deepcopy

E3 = np.ones((3, 3))


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
title = ['Imagem Original', 'Esqueleto da Imagem', 'Resultado Final']
ml.show_images(g, 1, title)
