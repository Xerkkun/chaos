import itertools
import numpy as np

color = 3
ancho = 500
alto = 250
matriz = np.zeros((color,ancho,alto),dtype=int)
i,j,k = list(range(0,3)),list(range(0,ancho)),list(range(0,alto))
f = 0
print(matriz.shape)
for element in itertools.product(i,j,k):
    matriz[element] = 2
#print(type(matriz))
