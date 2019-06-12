# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""


import numpy as np
from PIL import Image, ImageDraw, ImageFont
import MyLib as ml
from copy import deepcopy

def findPoints(img, value=255):
    shape = img.shape
    points = []
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if img[i, j] == value:
                points.append((j, i))

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
    a = min(points)
    points.remove(a)
    sorted = []
    sorted.append(a)

    for i in range(len(points)):
        minDist = 1000000000000
        idx = -1
        for idx, (x, y) in enumerate(points):
            xp, yp = sorted[i]
            d = np.sqrt((xp - x)**2 + (yp - y)**2)
            if d < minDist:
                minDist = d
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

def connectPoints(sortedPoints, img, threshold=25, closed=True):
    # 1. Digamos que P seja uma sequência de pontos ordenados, distintos, de valor 1 em uma imagem
    # binária. Especificamos dois pontos de partida, A e B. Estes são os dois vértices iniciais do polígono.
    # 2.Estabelecemos um limiar, T, e duas pilhas vazias, ABERTA e FECHADA.

    openStack = []
    closeStack = []

    initialA, initialB = 0, 17

    a = sortedPoints[initialA]
    b = sortedPoints[initialB]
    # 3. Se os pontos em P correspondem a uma curva fechada, colocamos A em ABERTA e B em ABERTA e
    # em FECHADA. Se os pontos correspondem a uma curva aberta, colocamos A em ABERTA e B em FECHADA.

    if closed:
        openStack.append(b)

    openStack.append(a)
    closeStack.append(b)

    lcPoint = closeStack[-1]   # close stack last point
    loPoint = openStack[-1]    # open stack last point

    while len(openStack) > 0:
        # 4. Calculamos os parâmetros da reta que passa pelo último vértice em FECHADA e pelo último vértice
        # em ABERTA.
        loPoint = openStack[-1]
        eq = getLineEquation(lcPoint, loPoint)
        # copy = deepcopy(img)
        # draw = ImageDraw.Draw(copy)
        # print("LCpoint {} LOpoint {}".format(lcPoint, loPoint))
        # draw.line([lcPoint, loPoint], fill=255, width=5)
        # copy.show()
        # input()
        lcIndex = sortedPoints.index(lcPoint)
        loIndex = sortedPoints.index(loPoint)

        distMax = 0
        idxMax = 0

        if loIndex > lcIndex:
            sublist = sortedPoints[loIndex:]
        else:
            sublist = sortedPoints[loIndex:lcIndex]

        for point in sublist:
            # 5. Calculamos as distâncias em relação a reta calculada na Etapa 4 para todos os pontos em P cuja
            # sequência os coloca entre os vértices da Etapa 4. Selecionamos o ponto, V máx , com a distância máxima,
            # D máx (os empates são resolvidos arbitrariamente).
            distance = distPoint2Line(eq, point)

            if distance > distMax:
                distMax = distance                        # getting max value
                idxMax = sortedPoints.index(point)

        if distMax > threshold:
            # 6. Se D máx > T, pomos V máx no final da pilha ABERTA
            # como um novo vértice. Vá para a Etapa 4.
            openStack.append(sortedPoints[idxMax])
        else:
            # 7. Se não, remova o último vértice de ABERTA e o
            # insira como o último vértice de FECHADA.
            closeStack.append(openStack.pop())
            lcPoint = closeStack[-1]

        # 8. Se ABERTA não estiver vazia, vamos para a Etapa 4.
        # 9. Caso contrário, saímos. Os vértices em FECHADA são os vértices do ajuste poligonal dos pontos pertencentes a P.
        print("ABERTA:")
        p = []
        for i in openStack:
            p.append(sortedPoints.index(i))
        print(p)
        p = []
        print("FECHADA:")
        for i in closeStack:
            p.append(sortedPoints.index(i))
        print(p)
        print(eq)
        print("")
    return closeStack


img = Image.open('images/pontos.bmp')
img = np.array(img)


points = findPoints(img, 255)
sorted = sortPoints(points)

# img = Image.fromarray(img)

f = connectPoints(sorted, img)

# ml.show_images([img])
img = Image.fromarray(img)
draw = ImageDraw.Draw(img)
fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
draw.line(f, fill=255, width=1)
for i, p in enumerate(sorted):
    draw.text(p, str(i), font=fnt, fill=(255))
img.show()
# print(f)
