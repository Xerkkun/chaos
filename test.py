import numpy as np
import math
import random
#===============================================================================
#Funciones
#===============================================================================
#Ecuaciones del oscilador HO1
def Fx1(x,y,z,w)
    a,b = 4., 6.
    Fx = a*x - b*y*z - 10.
    return Fx

def Fx2(x,y,z,w)
    c,k = 10.,2.5
    Fy = -c*y + x*z + k*w
    return Fy

def Fx3(x,y,z,w)
    d,e = 5.,2.
    Fz = -d*z + e*x*z
    return Fz

def Fx4(x,y,z,w)
#===============================================================================
#Métodos numéricos
def forward_euler(xn,yn,zn,wn,h)
    x1(i+1) = x1(i) + h*Fx1(x1(i),x2(i),x3(i),x4(i))
    x2(i+1) = x2(i) + h*Fx2(x1(i),x2(i),x3(i),x4(i))
    x3(i+1) = x3(i) + h*Fx3(x1(i),x2(i),x3(i),x4(i))
    x4(i+1) = x4(i) + h*Fx4(x1(i),x2(i),x3(i),x4(i))
return x,y,z,w
#===============================================================================
#Inicio

#parámetros de entrada
# n: pasos por corridas (1e6)
# s: corridas (1000)
# t: transitorio (5000)
# met: método numérico de resolución (FE,BE,RK4,AB6,AM4,G4)
# bin: método de generación de secuencias binarias (umbral,mod255)

nn,ss,tt,met,v,b = input('Parámetros de entrada: ').split()
n = int(float(nn))
s = int(ss)
t = int(tt)

print("Número de pasos:", n)
print("Número de corridas:", s)
print("Estado transitorio:", t)
print("Método numérico: ", met)
print("Variable para sec. binarias: ", v)
print("Metodo sec. binarias: ", b)

h = (0.001,0.01,0.001,0.001,0.005,0.005) #Ancho de paso para cada método
arch = open(b + "_" + met)

for i in range(1:n+1)

    if met == 'FE':
        x,y,z,w = forward_euler(xn,yn,zn,wn,h[0])
        arch.write(t,x,y,z,w)

arch.close
