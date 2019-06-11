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


def findPoints(img, value=255):
    shape = img.shape
    points = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if img[i, j] == value:
                points.append((i, j))

    return points


def getLineEquation(p1, p2):
    p1 = np.array(p1)   # [x, y]
    x1, y1 = p1[0], p1[1]

    p2 = np.array(p2)
    x2, y2 = p2[0], p2[1]

    p = p1 - p2         # [x1 - x2, y1 - y2]
    dx, dy = p[0], p[1]

    return np.array([dy, -dx, (x1*y2-x2*y1)])


def distPoint2Line(lineEquation, point):
    a, b, c = lineEquation[0], lineEquation[1], lineEquation[2]
    xp, yp = point[0], point[1]

    return np.abs(a*xp + b*yp + c) / np.sqrt(a**2 + b**2)


def sortPoints(points):
    a = points.pop(0)
    sorted = []
    sorted.append(a)

    for i in range(len(points)):
        min = 1000000000000
        idx = -1
        for idx, (x, y) in enumerate(points):
            xp, yp = sorted[i]
            d = np.sqrt((xp - x)**2 + (yp - y)**2)
            if d < min:
                min = d
                index = idx

        if index != -1:
            sorted.append(points.pop(index))
        else:
            print("error")

    return sorted


# from itertools import cycle
# lst = ['a', 'b', 'c']
# pool = cycle(lst)
# for item in pool:
#     print item,

def ligaPontos(pontos, limiar=20, closed=True):
    # 1. Digamos que P seja uma sequência de pontos ordenados, distintos, de valor 1 em uma imagem
    # binária. Especificamos dois pontos de partida, A e B. Estes são os dois vértices iniciais do polígono.
    # 2.Estabelecemos um limiar, T, e duas pilhas vazias, ABERTA e FECHADA.

    aberta = []
    fechada = []

    startA, startB = 0, 20

    a = pontos[startA]
    b = pontos[startB]
    # 3. Se os pontos em P correspondem a uma curva fechada, colocamos A em ABERTA e B em ABERTA e
    # em FECHADA. Se os pontos correspondem a uma curva aberta, colocamos A em ABERTA e B em FECHADA.

    if closed:
        aberta.append(b)

    aberta.append(a)
    fechada.append(b)

    while len(aberta) > 0:
        # 4. Calculamos os parâmetros da reta que passa pelo último vértice em FECHADA e pelo último vértice
        # em ABERTA.
        p1 = fechada[-1]
        p2 = aberta[-1]
        eq = getLineEquation(p1, p2)
        distances = []

        for i in range(pontos.index(p1)):
            # 5. Calculamos as distâncias em relação a reta calculada na Etapa 4 para todos os pontos em P cuja
            # sequência os coloca entre os vértices da Etapa 4. Selecionamos o ponto, V máx , com a distância máxima,
            # D máx (os empates são resolvidos arbitrariamente).
            distances.append(distPoint2Line(eq, pontos[i]))

        if len(distances) > 0:
            distMax = max(distances)    # getting max value
            idxMax = distances.index(distMax)

            if distMax > limiar:
                # 6. Se D máx > T, pomos V máx no final da pilha ABERTA
                # como um novo vértice. Vá para a Etapa 4.
                aberta.append(pontos[idxMax])
            else:
                # 7. Se não, remova o último vértice de ABERTA e o
                # insira como o último vértice de FECHADA.
                fechada.append(aberta.pop())

        # 8. Se ABERTA não estiver vazia, vamos para a Etapa 4.
        # 9. Caso contrário, saímos. Os vértices em FECHADA são os vértices do ajuste poligonal dos pontos pertencentes a P.
        print(len(fechada))
    return fechada


img = Image.open('images/pontos.bmp')
img = np.array(img)

points = findPoints(img, 255)
sorted = sortPoints(points)
f = ligaPontos(sorted)
print(f)
