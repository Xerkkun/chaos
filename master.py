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
#Métodos de secuencias binarias
def umbral(x,y,z,w,sel):
    u = (0,0,0,0)
    var2 = (x,y,z,w)

    if var2[sel] > u[sel]:
        bin = '1'
    else:
        bin = '0'
    return bin
#===============================================================================
#Inicio

#parámetros de entrada
# n: pasos por corridas (1e6)
# s: corridas (1000)
# t: transitorio (5000)
# met: método numérico de resolución (FE,BE,RK4,AB6,AM4,G4)
# bin: método de generación de secuencias binarias (umbral,mod255)

nn,ss,tt,met,v,b = input('Parámetros de entrada: ').split()
n,s,t = int(float(nn)),int(ss),int(tt)
nt = (n+t)*s #pasos totales

xo,yo,wo,zo = np.zeros(s+1,dtype=float),np.zeros(s+1,dtype=float),np.zeros(s+1,dtype=float),np.zeros(s+1,dtype=float)
c = (50,30,30,2)

for j in range(0,s+1):
    xo[j] = (random.random()-0.5)*c[0]
    yo[j] = (random.random()-0.5)*c[1]
    zo[j] = (random.random()-0.5)*c[2]
    wo[j] = (random.random()-0.5)*c[3]

print("Número de pasos:", n)
print("Número de corridas:", s)
print("Estado transitorio:", t)
print("Método numérico: ", met)
print("Variable para sec. binarias: ", v)
print("Metodo sec. binarias: ", b)

hh = (0.001,0.01,0.001,0.001,0.005,0.005) #Ancho de paso para cada método
arch = open(b + "_" + met + v + ".rnd","wb") #"wb" para escribir archivos con formato binario
r,i = -1,-1

var = ['x','y','z','w']
sel = var.index(v)

x,y,z,w = 0,0,0,0 #condiciones iniciales

if b == "umbral":
    k = 1
elif b == "mod255":
    k = 8

while r < s:
    i = i + 1

    if i==0:
        r = r + 1
        xn,yn,zn,wn = xo[r],yo[r],zo[r],wo[r]

    else:
        xn,yn,zn,wn = x,y,z,w

    if met == 'FE':
        h = hh[0]
        x,y,z,w = forward_euler(xn,yn,zn,wn,h)

    if abs(x)>50:
        print("Overflow in r = ",r)
        antes = arch.tell()
        xo[r] = (random.random()-0.5)*c[0]
        yo[r] = (random.random()-0.5)*c[1]
        zo[r] = (random.random()-0.5)*c[2]
        wo[r] = (random.random()-0.5)*c[3]
        pos = ((n*r)+r)-antes
        arch.seek(pos,1)
        despues = arch.tell()
        i = -1
        r = r - 1

    if i > (t-1):
        bin = umbral(x,y,z,w,sel)
        arch.write(bin.encode())
        if i == (n+t-1):
            if (r < s-1):
                arch.write(("\n").encode())
            i = -1

    #arch.write('%.5f' % t + '\t' + '%.5f' % x + '\t' + '%.5f' % y + '\t' + '%.5f' % z + '\t' + '%.5f' % w + '\n')


arch.close
