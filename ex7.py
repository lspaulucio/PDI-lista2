# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image, ImageDraw
import MyLib as ml
from copy import deepcopy
import matplotlib.pyplot as plt

img = Image.open('images/Fig11.10.jpg')
img = np.array(img)

border = ml.conv2D(img, ml.LAPLACE_FILTER)
Image.fromarray(border).show()
vertices = [(i, j) for i in range(img.shape[0]) for j in range(img.shape[1]) if border[i, j] == 255]
vertices = np.array(vertices)

gridSize = 40
grid = [(i, j) for i in range(0, img.shape[0]+1, gridSize) for j in range(0, img.shape[1]+1, gridSize)]
grid = np.array(grid)

points = []
for v in vertices:
    d = np.sqrt(np.square(grid - v).sum(axis=1))
    points.append(grid[np.argmin(d)])

points = np.array(points)
points = np.unique(points, axis=0)
points = [(j, i) for i, j in points]
result = Image.new('L', (img.shape[1], img.shape[0]))
d = ImageDraw.Draw(result)
print(len(points))
d.point(points, fill=255)

result.show()
