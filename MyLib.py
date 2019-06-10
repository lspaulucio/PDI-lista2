# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 1 - Pós-Graduação
    Data: 09/05/19
"""

import copy
import numpy as np
import matplotlib.pyplot as plt


def show_images(images, cols=1, titles=None, axis=False, interpolation='gaussian'):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if not axis:
            plt.axis("off")
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image, interpolation=interpolation)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def calculaHistograma(img):
    shape = img.shape
    histograma = np.zeros(256)
    x = [i for i in range(0, 256)]
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            pos = int(img[i][j])
            if pos < 0:
                pos = 0
            elif pos > 255:
                pos = 255

            histograma[pos] += 1

    return x, histograma


def equalizaHistograma(image):
    img = copy.deepcopy(image)
    x, histograma = calculaHistograma(img)

    hist_eq = np.zeros(256)
    hist_eq[0] = histograma[0]

    for i in range(1, 256):
        hist_eq[i] = hist_eq[i-1] + histograma[i]

    h = (hist_eq - np.min(hist_eq))/(np.max(hist_eq) - np.min(hist_eq))*255

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pos = int(img[i][j])
            if pos < 0:
                pos = 0
            elif pos > 255:
                pos = 255
            img[i][j] = h[pos]

    return img


def MSE(imgOrig, imgNoise):
    return np.square(imgOrig.astype(np.float) - imgNoise.astype(np.float)).mean()


def PSNR(imgOrig, imgNoise):
    mse = MSE(imgOrig, imgNoise)
    return 20 * np.log10(255/np.sqrt(mse))


def binarizaImg(img, threshold):
    newImg = copy.deepcopy(img)
    shape = newImg.shape
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if (newImg[i][j] > threshold):
                newImg[i][j] = 255
            else:
                newImg[i][j] = 0

    return newImg


def getOutputSize(old_shape, kernel_shape, padding, stride):
    return int((old_shape[0] + 2*padding - kernel_shape[0])/stride + 1), \
           int((old_shape[1] + 2*padding - kernel_shape[1])/stride + 1)


def getPaddingSize(shape, kernel_shape, stride=1):
    return ((shape[0] - 1)*stride + kernel_shape[0] - shape[0]) // 2


def getNewSize(shape, padding):
    return int((shape + 2*padding))


def paddingImage(img, padding_size=1, type='zeros'):
    shape = img.shape
    shape = (shape[0]+2*padding_size, shape[1]+2*padding_size)

    if type == 'ones':
        new = np.full(shape, 255)
    else:
        new = np.zeros(shape)

    new[padding_size:-padding_size, padding_size:-padding_size] = img

    if type == 'repeat':
        for i in range(padding_size):
            new[i, padding_size:-padding_size] = img[0]
            new[-i-1, padding_size:-padding_size] = img[-1]

        for i in range(padding_size):
            new[padding_size:-padding_size, i] = img[:, 0]
            new[padding_size:-padding_size, -i-1] = img[:, -1]

        for i in range(padding_size):
            for j in range(padding_size):
                new[i, j] = img[0, 0]           # upper left
                new[-i-1, -j-1] = img[-1, -1]   # lower right
                new[-i-1, j] = img[-1, 0]       # lower left
                new[i, -j-1] = img[0, -1]       # upper right

    return new


def conv2D(img, kernel, stride=1, padding=1, padding_type='zeros'):
    shape = []
    paddedImg = []
    newImg = []
    for i in range(0, 2):
        size = getNewSize(img.shape[i], padding)
        shape.append(size)

    kernel = np.flip(kernel)
    paddedImg = paddingImage(img, padding, padding_type)
    newImg = np.zeros((getOutputSize(img.shape, kernel.shape, padding, stride)))

    x, y = kernel.shape[0], kernel.shape[1]

    for i in range(0, newImg.shape[0]):
        for j in range(0, newImg.shape[1]):
            value = (kernel * paddedImg[i:i+y, j:j+x]).sum()
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            newImg[i][j] = value

    return newImg


def medianFilter(img, kernel_shape=(3, 3), padding=1, padding_type='zeros'):
    newImg = np.zeros(img.shape)
    paddedImg = paddingImage(img, padding, type=padding_type)
    x, y = kernel_shape[0], kernel_shape[1]

    for i in range(0, newImg.shape[0]):
        for j in range(0, newImg.shape[1]):
            value = np.median(paddedImg[i:i+y, j:j+x])
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            newImg[i][j] = value

    return newImg


def adaptativeMedian(img, Smax):
    window_size = 3  # window initial size
    padding_size = getPaddingSize(img.shape, Smax)
    paddedImg = paddingImage(img, padding_size, 'repeat')
    newImg = np.zeros(img.shape)
    x, y = 0, 0
    for i in range(padding_size, paddedImg.shape[0]-padding_size):
        for j in range(padding_size, paddedImg.shape[1]-padding_size):
            temp_size = window_size
            stepA = True
            value = 0
            while(stepA):
                s = temp_size // 2
                window = paddedImg[i-s:i+s+1, j-s:j+s+1]
                zmed = np.median(window)
                zmin = np.min(window)
                zmax = np.max(window)
                zxy = paddedImg[i, j]
                # StepA
                a1 = zmed - zmin                 # a1 = zmed -zmin
                a2 = zmed - zmax                 # a2 = zmed - zmax
                if a1 > 0 and a2 < 0:            # se a1 > 0 e a2 <0 goto step B
                    # stepB
                    b1 = zxy - zmin              # b1 = zxy - zmin
                    b2 = zxy - zmax              # b2 = zxy - zmax
                    if b1 > 0 and b2 < 0:        # se b1 > 0 e b2 < 0
                        value = zxy              # saida e zxy
                        stepA = False
                    else:
                        value = zmed             # senao saida e zmed
                        stepA = False
                else:
                    temp_size += 2               # senao aumente sxy
                    if temp_size <= Smax[0]:  # se window_size <= smax repita A
                        stepA = True
                    else:
                        value = zmed             # senao saida e zmed
                        stepA = False

            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            newImg[x][y] = value

            y += 1
        x += 1
        y = 0

    return newImg


# ############################# Filters ################################

MEAN_FILTER = np.ones((3, 3)) / 9.0

MEAN_FILTER_11 = np.ones((11, 11)) / (11*11)

MEAN_FILTER2 = 1/16. * np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]])

LAPLACE_FILTER2 = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])

LAPLACE_FILTER = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

SOBEL_FILTER = np.array([[1,   2,  1],
                         [0,   0,  0],
                         [-1, -2, -1]])


# ############################ Frequence domain functions

def frequenceSpace(img, center):
    D = np.zeros(img.shape)
    for u in range(0, img.shape[0]):
        for v in range(0, img.shape[1]):
            D[u][v] = np.sqrt((u - center[0])**2 + (v - center[1])**2)
    return D


def fshift(img):
    for u in range(0, img.shape[0]):
        for v in range(0, img.shape[1]):
            img[u][v] *= (-1)**(u+v)
    return img


def extendImg(img):
    newShape = img.shape[0]*2, img.shape[1]*2
    I_bg = np.zeros(newShape)
    I_bg[0:img.shape[0], 0:img.shape[1]] = img
    return I_bg


def imgFilter(img, D0=30, n=1, type='btw', center=None):
    if center is None:
        center = (img.shape[1]/2, img.shape[0]/2)

    y, x = center
    D = frequenceSpace(img, (x, y))
    H = np.zeros(img.shape)
    W = 50
    if type == 'btw':
        H = 1./(1+(D/D0)**(2*n))
    elif type == 'gaus':
        H = 1 - np.exp(-((D**2 - D0**2)/(D*W+1))**2)
    elif type == 'ideal':
        H[D <= D0] = 1
    return H


def filterHomomorphic(img, D0=30, n=1, center=None, yh=2, yl=0.5, c=1):
    if center is None:
        center = (img.shape[1]/2, img.shape[0]/2)

    y, x = center
    D = frequenceSpace(img, (x, y))
    H = np.zeros(img.shape)
    H = (yh - yl) * (1 - np.exp(-c*(D**2)/(D0**2))) + yl
    return H


def notchFilter(img, D0, n=1, type='btw', center=None):
    if center is None:
        center = (img.shape[1]/2, img.shape[0]/2)

    y, x = center
    D = frequenceSpace(img, (x, y))
    if type == "btw":
        H = 1./(1+(D/D0)**(2*n))
    elif type == "gaus":
        H = np.exp(-D**2/(2*D0**2))  # gaussiano LPF
    elif type == "ideal":
        H = np.zeros(D.shape)
        H[D <= D0] = 1

    return 1 - H
