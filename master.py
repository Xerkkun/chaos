import numpy as np
import math
import random
#===============================================================================
#Funciones
#===============================================================================
#Ecuaciones del oscilador HO1
def Fx1(x,y,z,w):
    a,b = 4., 6.
    Fx = (a*x) - (b*y*z) - 10.
    return Fx

def Fx2(x,y,z,w):
    c,k = 10.,2.5
    Fy = (-c*y) + (x*z) + (k*w)
    return Fy

def Fx3(x,y,z,w):
    d,e = 5.,2.
    Fz = (-d*z) + (e*x*y)
    return Fz

def Fx4(x,y,z,w):
    f = 0.05
    Fw = f*(x+z)
    return Fw
#===============================================================================
#Métodos numéricos
def forward_euler(xn,yn,zn,wn,h):
    x,y,z,w = xn+h*Fx1(xn,yn,zn,wn),yn+h*Fx2(xn,yn,zn,wn),zn+h*Fx3(xn,yn,zn,wn),wn+h*Fx4(xn,yn,zn,wn)
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

xo,yo,wo,zo = np.zeros(s,dtype=float),np.zeros(s,dtype=float),np.zeros(s,dtype=float),np.zeros(s,dtype=float)
c = (60,30,15,)

for j in range(0,s):
    xo[j] = (random.random()-0.5)*60
    yo[j] = (random.random()-0.5)*30
    zo[j] = (random.random()-0.5)*40
    wo[j] = (random.random()-0.5)*2

print("Número de pasos:", n)
print("Número de corridas:", s)
print("Estado transitorio:", t)
print("Método numérico: ", met)
print("Variable para sec. binarias: ", v)
print("Metodo sec. binarias: ", b)

hh = (0.001,0.01,0.001,0.001,0.005,0.005) #Ancho de paso para cada método
arch = open(b + "_" + met + v + ".rnd","w") #"wb" para binario
o = 0
i = 0
t = 0

x,y,z,w = -0.91,-2.3,0.91,1.#condiciones iniciales

while i < n+1:
#     if i%s == 0:
#         xn,yn,zn,wn = xo[o],yo[o],zo[o],wo[o]
#         o = o + 1
#     else:
#         xn,yn,zn,wn = x,y,z,w

    arch.write('%.5f' % t + '\t' + '%.5f' % x + '\t' + '%.5f' % y + '\t' + '%.5f' % z + '\t' + '%.5f' % w + '\n')
    h = hh[0]
    xn,yn,zn,wn = x,y,z,w

    x = xn+(h*Fx1(xn,yn,zn,wn))
    y = yn+(h*Fx2(xn,yn,zn,wn))
    z = zn+(h*Fx3(xn,yn,zn,wn))
    w = wn+(h*Fx4(xn,yn,zn,wn))

    i = i + 1
    t = t + h

arch.close
