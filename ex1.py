# -*- coding: utf-8 -*-

""" Processamento Digital de Imagens
    Aluno: Leonardo Santos Paulucio
    Lista de Exercicios 2 - Pós-Graduação
    Data: 23/06/19
"""

from PIL import Image
import numpy as np


img = Image.open('images/estrada.png')              # Le a imagem
img.convert('RGB')
img = np.array(img)                                 # Converte a imagem para um numpy array

print(img[0,0])
