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
#-------------------------------------------------------------------------------
def backward_euler(xn,yn,zn,wn,h):
    xn1,yn1,zn1,wn1 = forward_euler(xn,yn,zn,wn,h)
    x,y,z,w = xn+h*Fx1(xn1,yn1,zn1,wn1),yn+h*Fx2(xn1,yn1,zn1,wn1),zn+h*Fx3(xn1,yn1,zn1,wn1),wn+h*Fx4(xn1,yn1,zn1,wn1)
    return x,y,z,w
#-------------------------------------------------------------------------------
def runge_kutta4(xn,yn,zn,wn,h):
    k1 = h*Fx1(xn,yn,zn,wn)
    k2 = Fx1(xn + h*0.5*k1,yn + h*0.5*k1,zn + h*0.5*k1,wn + h*0.5*k1)
    k3 = Fx1(xn + h*0.5*k2,yn + h*0.5*k2,zn + h*0.5*k2,wn + h*0.5*k2)
    k4 = Fx1(xn + h*k3,yn + h*k3,zn + h*k3,wn + h*k3)
    x = xn + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)

    k1 = h*Fx2(xn,yn,zn,wn)
    k2 = Fx2(xn + h*0.5*k1,yn + h*0.5*k1,zn + h*0.5*k1,wn + h*0.5*k1)
    k3 = Fx2(xn + h*0.5*k2,yn + h*0.5*k2,zn + h*0.5*k2,wn + h*0.5*k2)
    k4 = Fx2(xn + h*k3,yn + h*k3,zn + h*k3,wn + h*k3)
    y = yn + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)

    k1 = h*Fx3(xn,yn,zn,wn)
    k2 = Fx3(xn + h*0.5*k1,yn + h*0.5*k1,zn + h*0.5*k1,wn + h*0.5*k1)
    k3 = Fx3(xn + h*0.5*k2,yn + h*0.5*k2,zn + h*0.5*k2,wn + h*0.5*k2)
    k4 = Fx3(xn + h*k3,yn + h*k3,zn + h*k3,wn + h*k3)
    z = zn + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)

    k1 = h*Fx4(xn,yn,zn,wn)
    k2 = Fx4(xn + h*0.5*k1,yn + h*0.5*k1,zn + h*0.5*k1,wn + h*0.5*k1)
    k3 = Fx4(xn + h*0.5*k2,yn + h*0.5*k2,zn + h*0.5*k2,wn + h*0.5*k2)
    k4 = Fx4(xn + h*k3,yn + h*k3,zn + h*k3,wn + h*k3)
    w = wn + (h/6.)*(k1 + 2*k2 + 2*k3 + k4)

    return x,y,z,w
#===============================================================================
#===============================================================================
#Inicio

#parámetros de entrada
# n: pasos por corridas (1e6)
# met: método numérico de resolución (FE,BE,RK4,AB6,AM4,G4)

nn,met = input('Parámetros de entrada: ').split()
n = int(float(nn))

c = (50,30,30,2)

xo = (random.random()-0.5)*c[0]
yo = (random.random()-0.5)*c[1]
zo = (random.random()-0.5)*c[2]
wo = (random.random()-0.5)*c[3]

print("Número de pasos:", n)
print("Método numérico: ", met)

hh = (0.001,0.01,0.001,0.001,0.005,0.005) #Ancho de paso para cada método
arch = open(met + ".rnd","w") #"wb" para escribir archivos con formato binario
i,t = 0,0

x,y,z,w = xo,yo,zo,wo #condiciones iniciales

for i in range(0,n):
    xn,yn,zn,wn = x,y,z,w

    if met == 'FE':
        h = hh[0]
        x,y,z,w = forward_euler(xn,yn,zn,wn,h)
    elif met == 'BE':
        h = hh[1]
        x,y,z,w = backward_euler(xn,yn,zn,wn,h)
    elif met == 'RK4':
        h = hh[2]
        x,y,z,w = runge_kutta4(xn,yn,zn,wn,h)

    arch.write('%.5f' % t + '\t' + '%.5f' % x + '\t' + '%.5f' % y + '\t' + '%.5f' % z + '\t' + '%.5f' % w + '\n')
    t = t + h

arch.close
