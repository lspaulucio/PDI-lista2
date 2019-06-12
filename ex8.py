# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image
import MyLib as ml


def getMoment(img, p, q):
    shape = img.shape

    sum = 0
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            sum += (x**p) * (y**q) * img[x, y]

    return sum


def getCentralMoment(img, p, q, x_bar=None, y_bar=None):

    shape = img.shape

    sum = 0
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            sum += (x - x_bar)**p * (y - y_bar)**q * img[x, y]

    return sum


def getGama(p, q):
    if p + q >= 2:
        return 1 + (p+q)/2
    else:
        print("Erro gama")


def getNormalizedCentralMoment(p, q, upq=None, u00=None):

    gama = getGama(p, q)

    return upq / (u00**gama)


def getInvariantMoments(img):
    invariantMoments = np.zeros(7)

    m00 = getMoment(img, 0, 0)  # M00
    m01 = getMoment(img, 0, 1)  # M01
    m10 = getMoment(img, 1, 0)  # M10

    x_bar = m10 / m00
    y_bar = m01 / m00

    u00 = getCentralMoment(img, 0, 0, x_bar, y_bar)
    u02 = getCentralMoment(img, 0, 2, x_bar, y_bar)
    u03 = getCentralMoment(img, 0, 3, x_bar, y_bar)
    u11 = getCentralMoment(img, 1, 1, x_bar, y_bar)
    u12 = getCentralMoment(img, 1, 2, x_bar, y_bar)
    u20 = getCentralMoment(img, 2, 0, x_bar, y_bar)
    u21 = getCentralMoment(img, 2, 1, x_bar, y_bar)
    u30 = getCentralMoment(img, 3, 0, x_bar, y_bar)

    n02 = getNormalizedCentralMoment(0, 2, u02, u00)
    n03 = getNormalizedCentralMoment(0, 3, u03, u00)
    n11 = getNormalizedCentralMoment(1, 1, u11, u00)
    n12 = getNormalizedCentralMoment(1, 2, u12, u00)
    n20 = getNormalizedCentralMoment(2, 0, u20, u00)
    n21 = getNormalizedCentralMoment(2, 1, u21, u00)
    n30 = getNormalizedCentralMoment(3, 0, u30, u00)

    invariantMoments[0] = n20 + n02
    invariantMoments[1] = (n20 - n02)**2 + 4*(n11**2)
    invariantMoments[2] = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    invariantMoments[3] = (n30 + n12)**2 + (n21 + n03)**2
    invariantMoments[4] = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + (3*n21 - n03)*(n21 + n03) * (3 * (n30 + n12)**2 - (n21 + n03)**2)
    invariantMoments[5] = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30+n12)*(n21+n03)
    invariantMoments[6] = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2-3*(n21 + n03)**2)-(n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2-(n21 + n03)**2)

    return invariantMoments


# a) imagem normal;
imgOriginal = Image.open('images/lena.tif')
img = np.array(imgOriginal)

# b) imagem redimensionada a metade,
newSize = imgOriginal.size[0]//2, imgOriginal.size[1]//2
resizedImage = imgOriginal.resize(newSize)
resizedImage = np.array(resizedImage)
# c) rotacionada de 90º;
rotatedImage = imgOriginal.rotate(90)
rotatedImage = np.array(rotatedImage)
# d) rotacionada de 180º
flipImage = imgOriginal.rotate(180)
flipImage = np.array(flipImage)
# print(imgOriginal.size)
g = [img, resizedImage, rotatedImage, flipImage]
# print(getInvariantMoments(img))
print("Original: {}".format(getInvariantMoments(img)))
print("Resized: {}".format(getInvariantMoments(resizedImage)))
print("Rotated: {}".format(getInvariantMoments(rotatedImage)))
print("Flip: {}".format(getInvariantMoments(flipImage)))
