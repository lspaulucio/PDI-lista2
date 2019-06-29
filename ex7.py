# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 30/06/19
"""


import numpy as np
import MyLib as ml
from copy import deepcopy
from PIL import Image, ImageDraw


def getPoints(img):
    vertices = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1]) if img[i, j] == 255]
    vertices = np.array(vertices)
    return vertices


def makeGrid(img, gridSize):
    grid = [(i, j) for i in range(0, img.shape[0], gridSize) for j in range(0, img.shape[1], gridSize)]
    return grid


def convert2grid(vertices, grid):
    points = []
    grid = np.array(grid)
    for v in vertices:
        d = np.sqrt(np.square(grid - v).sum(axis=1))
        points.append(grid[np.argmin(d)])

    points = np.array(points)
    points = np.unique(points, axis=0)
    points = [(j, i) for i, j in points]
    return points


def getStartPoint(points):
    max, idx = 0, 0
    for i, (y, x) in enumerate(points):
        if y > max:
            max = y
            idx = i

    return points[idx], idx


def getNextPoint(currentPoint, freemanIdx, gridSize):
    if freemanIdx == 0:
        return currentPoint[0] + gridSize, currentPoint[1]
    elif freemanIdx == 1:
        return currentPoint[0] + gridSize, currentPoint[1] + gridSize
    elif freemanIdx == 2:
        return currentPoint[0], currentPoint[1] + gridSize
    elif freemanIdx == 3:
        return currentPoint[0] - gridSize, currentPoint[1] + gridSize
    elif freemanIdx == 4:
        return currentPoint[0] - gridSize, currentPoint[1]
    elif freemanIdx == 5:
        return currentPoint[0] - gridSize, currentPoint[1] - gridSize
    elif freemanIdx == 6:
        return currentPoint[0], currentPoint[1] - gridSize
    elif freemanIdx == 7:
        return currentPoint[0] + gridSize, currentPoint[1] - gridSize


def getFreemanChain(points, gridSize):
    startPoint, idx = getStartPoint(points)

    freemanChain = []

    currentPoint = startPoint
    sortedPoints = []
    # points.append(startPoint)

    while len(points) != 1:
        for i in range(8):
            nextPoint = getNextPoint(currentPoint, i, gridSize)
            if nextPoint in points:
                freemanChain.append(i)
                points.remove(currentPoint)
                sortedPoints.append(currentPoint)
                currentPoint = nextPoint
                break

    sortedPoints.append(currentPoint)
    sortedPoints.append(startPoint)
    return freemanChain, sortedPoints


img = Image.open('images/Fig11.10.jpg')
img = np.array(img)

# Getting object border
border = ml.conv2D(img, ml.LAPLACE_FILTER)

images = [deepcopy(border)]

# Getting border points
vertices = getPoints(border)

gridSize = 43
# Creating grid
grid = makeGrid(border, gridSize)

# Resampling border points to grid
points = convert2grid(vertices, grid)

# Finding Freeman chain
f, p = getFreemanChain(points, gridSize)

print("Grid size: {}".format(gridSize))
print("Freeman chain size: {}".format(len(f)))
print("Freeman chain: {}".format(f))

# Making results to show
result = Image.new('L', (img.shape[1], img.shape[0]))
border = Image.fromarray(border)
d = ImageDraw.Draw(border)
grid = [(j, i) for i, j in grid]
d.point(grid, fill=255)
images.append(deepcopy(np.array(border)))
d = ImageDraw.Draw(result)
d.point(p, fill=255)
images.append(deepcopy(np.array(result)))
d.line(p, fill=255)
images.append(deepcopy((np.array(result))))

titles = ['Borda externa', 'Imagem com o grid', 'Fronteira subamostrada', 'Pontos conectados']
ml.show_images(images, 1, titles)
