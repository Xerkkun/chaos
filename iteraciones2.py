import itertools
import numpy as np

matriz = np.zeros((3,500,250),dtype=int)
color,ancho,alto = list(range(0,3)),list(range(0,500)),list(range(0,250))

for i in color:
    for j in ancho:
        for k in alto:
            matriz[i,j,k] = 2
            print(matriz[i,j,k])
print(type(matriz))
