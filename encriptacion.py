import cv2
import numpy as np
import itertools
import math
import random
#===============================================================================
#Funciones
#===============================================================================
#Ecuaciones del oscilador HO3a
def Fx1(x,y,z,w):
    a = 16.
    Fx = a*(y-x) + (y*z) + w
    return Fx

def Fx2(x,y,z,w):
    b,c,p = 3.,8.,0.1
    Fy = b*y - c*x*z - p*x**2 + w
    return Fy

def Fx3(x,y,z,w):
    d = 20.
    Fz = x*y - d
    return Fz

def Fx4(x,y,z,w):
    Fw = -x-y
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
#-------------------------------------------------------------------------------
def adams_bashforth(
    xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,
    xn3,yn3,zn3,wn3,xn4,yn4,zn4,wn4,xn5,yn5,zn5,wn5,h):

    x = (xn + (h/1440.) * (4277*Fx1(xn,yn,zn,wn) - 7923*Fx1(xn1,yn1,zn1,wn1) +
    9982*Fx1(xn2,yn2,zn2,wn2)-7298*Fx1(xn3,yn3,zn3,wn3) + 2877*Fx1(xn4,yn4,zn4,wn4) -
    475*Fx1(xn5,yn5,zn5,wn5)))

    y = (yn + (h/1440.) * (4277*Fx2(xn,yn,zn,wn) - 7923*Fx2(xn1,yn1,zn1,wn1) +
    9982*Fx2(xn2,yn2,zn2,wn2)-7298*Fx2(xn3,yn3,zn3,wn3) + 2877*Fx2(xn4,yn4,zn4,wn4) -
    475*Fx2(xn5,yn5,zn5,wn5)))

    z = (zn + (h/1440.) * (4277*Fx3(xn,yn,zn,wn) - 7923*Fx3(xn1,yn1,zn1,wn1) +
    9982*Fx3(xn2,yn2,zn2,wn2)-7298*Fx3(xn3,yn3,zn3,wn3) + 2877*Fx3(xn4,yn4,zn4,wn4) -
    475*Fx3(xn5,yn5,zn5,wn5)))

    w = (wn + (h/1440.) * (4277*Fx4(xn,yn,zn,wn) - 7923*Fx4(xn1,yn1,zn1,wn1) +
    9982*Fx4(xn2,yn2,zn2,wn2)-7298*Fx4(xn3,yn3,zn3,wn3) + 2877*Fx4(xn4,yn4,zn4,wn4) -
    475*Fx4(xn5,yn5,zn5,wn5)))

    return x,y,z,w
#-------------------------------------------------------------------------------
def adams_moulton4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,h):

    x1,y1,z1,w1 = forward_euler(xn,yn,zn,wn,h)

    x = xn + (h/24.) * (9*Fx1(x1,y1,z1,w1) + 19*Fx1(xn,yn,zn,wn) -
    5*Fx1(xn1,yn1,zn1,wn1) + Fx1(xn2,yn2,zn2,wn2))

    y = yn + (h/24.) * (9*Fx2(x1,y1,z1,w1) + 19*Fx2(xn,yn,zn,wn) -
    5*Fx2(xn1,yn1,zn1,wn1) + Fx2(xn2,yn2,zn2,wn2))

    z = zn + (h/24.) * (9*Fx3(x1,y1,z1,w1) + 19*Fx3(xn,yn,zn,wn) -
    5*Fx3(xn1,yn1,zn1,wn1) + Fx3(xn2,yn2,zn2,wn2))

    w = wn + (h/24.) * (9*Fx4(x1,y1,z1,w1) + 19*Fx4(xn,yn,zn,wn) -
    5*Fx4(xn1,yn1,zn1,wn1) + Fx4(xn2,yn2,zn2,wn2))

    return x,y,z,w
#===============================================================================
def gear4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,xn3,yn3,zn3,wn3,h):
    x1,y1,z1,w1 = forward_euler(xn,yn,zn,wn,h)

    x = (48./25.)*xn - (36./25.)*xn1 + (16./25.)*xn2 - (3./25.)*xn3 + (12./25.)*h*Fx1(x1,y1,z1,w1)
    y = (48./25.)*yn - (36./25.)*yn1 + (16./25.)*yn2 - (3./25.)*yn3 + (12./25.)*h*Fx2(x1,y1,z1,w1)
    z = (48./25.)*zn - (36./25.)*zn1 + (16./25.)*zn2 - (3./25.)*zn3 + (12./25.)*h*Fx3(x1,y1,z1,w1)
    w = (48./25.)*wn - (36./25.)*wn1 + (16./25.)*wn2 - (3./25.)*wn3 + (12./25.)*h*Fx4(x1,y1,z1,w1)

    return x,y,z,w
#===============================================================================
#Métodos de secuencias binarias
def umbral(x,y,z,w,sel):
    u = (0,0,0,0)
    var2 = (x,y,z,w)

    if var2[sel] > u[sel]: bin = '1'
    else: bin = '0'
    return bin
#-------------------------------------------------------------------------------
def mod255(x,y,z,w,sel):
    u = (0,0,0,0)
    var2 = (x,y,z,w)
    bin = format(int((var2[sel]*1E6) % 255),'b')
    while len(bin) < 8 :	#Acompleta los numeros binarios a palabras de 8 bits
        bin = '0' + bin
    return bin
#-------------------------------------------------------------------------------
def xor(binp):
    pos_bin = 0
    for i in range(0,5):
        pos_bin = pos_bin ^ int(binp[i])
    return pos_bin
#-------------------------------------------------------------------------------
def xor_shift(binp):
    global A,B,C,D,E,F,regA,regB,regC,regD
    A = int(binp) ^ C
    B = int(regA[2]) ^ D
    C = int(regB[1]) ^ int(regB[2])
    D = int(regD[1]) ^ int(regD[2])
    E = D ^ A
    F = C ^ int(regC[2])
    regA = str(A) + regA[0:7]
    regB = str(B) + regB[0:7]
    regC = str(E) + regC[0:7]
    regD = str(F) + regD[0:7]
    out = str(E)
    return out
#===============================================================================
#Inicio

#parámetros de entrada
# n: pasos por corridas (1e6)
# s: corridas (1000)
# t: transitorio (5000)
# met: método numérico de resolución (FE,BE,RK4,AB6,AM4,G4)
# bin: método de generación de secuencias binarias (umbral,mod255)
# nstep: frecuencia de muestreo
# pos_pro: método para posprocesamiento (xor,xor_shift,gray)

image1 = cv2.imread("im1.jpg")
alto,ancho,color = image1.shape
nn,ss,tt,met,v,b,nstp,pos_pro = color*ancho*alto,1,5000,'FE','x','mod255',1,'xor_shift'
n,s,t,nstep = int(float(nn)),int(ss),int(tt),int(nstp)

xo,yo,wo,zo = 0.2,0.2,0.2,0.2

print("Número de pasos:", n)
print("Número de corridas:", s)
print("Estado transitorio:", t)
print("Método numérico: ", met)
print("Variable para sec. binarias: ", v)
print("Metodo sec. binarias: ", b)
print("Método para posprocesamiento", pos_pro)

salida = b + "_" + met + v + "_" + pos_pro + ".rnd"
hh = (0.001,0.01,0.001,0.001,0.005,0.005) #Ancho de paso para cada método
r,i = -1,-1
binp = ''
pos_bin = ''

var = ['x','y','z','w']
sel = var.index(v)

x,y,z,w = 0,0,0,0 #condiciones iniciales
xn1,yn1,zn1,wn1 = 0,0,0,0
xn2,yn2,zn2,wn2 = 0,0,0,0
xn3,yn3,zn3,wn3 = 0,0,0,0
xn4,yn4,zn4,wn4 = 0,0,0,0
xn5,yn5,zn5,wn5 = 0,0,0,0

if b == "umbral": k = 8
elif b == "mod255": k = 1
else: print("Método no definido")

if pos_pro == "xor": kk = 5
elif pos_pro == "xor_shift": kk = 1
else: print("Método no definido")

regA,regB,regC,regD = '000','000','000','000'
A,B,C,D,E,F = 0,0,0,0,0,0

x,y,z,w = xo,yo,zo,wo

for i in range (0,t): #Transitorio
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
    elif met == 'AB6':
        h = hh[3]
        x,y,z,w = adams_bashforth(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,
                                xn3,yn3,zn3,wn3,xn4,yn4,zn4,wn4,xn5,yn5,zn5,wn5,h)
    elif met == 'AM4':
        h = hh[4]
        x,y,z,w = adams_moulton4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,h)
    elif met == 'G4':
        h = hh[5]
        x,y,z,w = gear4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,xn3,yn3,zn3,wn3,h)
    else: print("Método no definido")

    xn5,yn5,zn5,wn5 = xn4,yn4,zn4,wn4
    xn4,yn4,zn4,wn4 = xn3,yn3,zn3,wn3
    xn3,yn3,zn3,wn3 = xn2,yn2,zn2,wn2
    xn2,yn2,zn2,wn2 = xn1,yn1,zn1,wn1
    xn1,yn1,zn1,wn1 = xn,yn,zn,wn

encryp = np.zeros((alto,ancho,color),dtype=int)
dim_i,dim_j,dim_k = list(range(0,alto)),list(range(0,ancho)),list(range(0,color))
salida1 = open('data1.dat','w')
salida2 = open('data2.dat','w')
f = 0

for element in itertools.product(dim_i,dim_j,dim_k):
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
    elif met == 'AB6':
        h = hh[3]
        x,y,z,w = adams_bashforth(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,
                                xn3,yn3,zn3,wn3,xn4,yn4,zn4,wn4,xn5,yn5,zn5,wn5,h)
    elif met == 'AM4':
        h = hh[4]
        x,y,z,w = adams_moulton4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,h)
    elif met == 'G4':
        h = hh[5]
        x,y,z,w = gear4(xn,yn,zn,wn,xn1,yn1,zn1,wn1,xn2,yn2,zn2,wn2,xn3,yn3,zn3,wn3,h)
    else: print("Método no definido")

    xn5,yn5,zn5,wn5 = xn4,yn4,zn4,wn4
    xn4,yn4,zn4,wn4 = xn3,yn3,zn3,wn3
    xn3,yn3,zn3,wn3 = xn2,yn2,zn2,wn2
    xn2,yn2,zn2,wn2 = xn1,yn1,zn1,wn1
    xn1,yn1,zn1,wn1 = xn,yn,zn,wn

    if b == "umbral": bin = umbral(x,y,z,w,sel)
    elif b == "mod255": bin = mod255(x,y,z,w,sel)
    binp = binp + bin

    if len(binp)==40 and pos_pro=="xor":
        for j in range(0,40,5):
            pos_bin = pos_bin + xor(binp[j:j+5])
            if len(pos_bin)==8:
                encryp[element] = int(pos_bin,2)
                pos_bin = ''
        binp = ''

    elif len(binp)==8 and pos_pro == "xor_shift":
        for j in range(0,8):
            pos_bin = pos_bin + xor_shift(binp[j])

    encryp[element] = int(pos_bin,2)
    pos_bin = ''
    binp = ''

cv2.imwrite('encryp.png',encryp)
encryp2 = cv2.imread("encryp.png")
xor1 = cv2.bitwise_xor(image1, encryp2)
cv2.imwrite('Xor1.png',xor1)
xor2 = cv2.bitwise_xor(xor1, encryp2)
cv2.imwrite('Xor2.png',xor2)
