# secuencial (no paralelizado)
# ==============================================================================
import pandas as pd
import numpy as np

# Se define la funcion
def suma_acumulada(number):
    return sum(range(1, number + 1))

# Lista de elementos sobre los que se quiere aplicar la funcion
valores = [10**8, 10**8, 10**8, 10**8, 10**8]

# Aplicar la funcion sobre cada elemento de forma secuencial
resultados = []

for valor in valores:
    resultado = suma_acumulada(valor)
    resultados.append(resultado)

resultados
