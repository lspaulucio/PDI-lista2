# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""

from PIL import Image
import numpy as np

img = Image.open('images/peppers.tiff')             # Le a imagem
img = np.array(img)                                 # Converte a imagem para um numpy array

chR = img[:, :, 0]
chG = img[:, :, 1]
chB = img[:, :, 2]

index = chR > (chG + 20)
index2 = chR > (chB + 20)

x = index & index2
print(x)
chR[x], chB[x] = chB[x], chR[x]

Image.fromarray(img).show()
