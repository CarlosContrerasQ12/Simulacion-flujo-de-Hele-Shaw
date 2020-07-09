import numpy as np
from scipy import fftpack as sp
from scipy import integrate as inte
from scipy import signal
from scipy import optimize as op
import matplotlib.pyplot as plt
import numba
from numba import jit
import warnings
import os
import time
warnings.filterwarnings("ignore")
os.chdir('datos/')


#Parametros iniciales
N=2048
alphas=np.arange(0,2*np.pi,2*np.pi/N,dtype='float64')
print("Numero de puntos en la discretizacion: ",np.size(alphas))
h=2*np.pi/N
dt=1e-5
Amu=1.0

s=np.fft.fftfreq(N)
k=s*N


@jit
def dife(zs):
	dfou=1.0j * k * np.fft.fft(zs)
	ins=np.where(np.abs(dfou)<1e-12)
	dfou[ins]=0
	return np.real(np.fft.ifft(dfou))
	#return sp.diff(zs)

@jit
def difeNoPer(zs):
	y1=np.cos(zs)
	y2=np.sin(zs)
	yp2=dife(y2)
	sg=np.sign(yp2)/np.sign(y1)
	return sg*np.sqrt(dife(y1)**2+yp2**2)
	
@jit	
def calcularW(gamma0,z,zal):
	W=1j*np.zeros(N,dtype='float64')
	for i in range(N):
		suma=0
		for j in range(N):
			if (abs(j-i))%2 ==1:
				suma+=(gamma0[j]/(z[i]-z[j]))	
		W[i]=-(1.0/z[i])+((1.0/(2*np.pi*1j))*suma*2*h)+((gamma0[i]*np.conj(zal[i]))/(2.0*(np.absolute(zal[i])**2)))
	return W
	
@jit
def gammaf(gammaa,z,zal):
	gammar=np.zeros(N,dtype='float64')
	for i in range(N):
		suma=0
		for j in range(N):
			if (abs(j-i))%2 ==1:
				suma+=(gammaa[j]/(z[i]-z[j]))	
		gammar[i]=2.0*Amu*np.real(-(zal[i]/z[i])+(zal[i]*(1.0/(2.0*np.pi*1j))*suma*2.0*h))
	return gammar
	
@jit
def punto_fijo_gamma(gamma0, z,zal):
	gammaa=gamma0
	for veces in range(17):
		for i in range(N):
			suma=0
			for j in range(N):
				if (abs(j-i))%2 ==1:
					suma+=(gammaa[j]/(z[i]-z[j]))	
			gammaa[i]=2.0*Amu*np.real(-(zal[i]/z[i])+zal[i]*(1.0/(2*np.pi*1j))*suma*2*h)
	return gammaa

	
x=np.cos(alphas)
y=np.sin(alphas)-0.1
z=x+1j*y

xd=dife(x)
yd=dife(y)
zal=xd+1j*yd


gammainicial=np.ones(N,dtype='float64')*0.9
ti=time.time()
gamma=op.fixed_point(gammaf,gammainicial,args=(z,zal))	
print(time.time()-ti)


zc1=dt*calcularW(gamma,z,zal)+np.conj(z)
z1=np.conj(zc1)
x1=np.real(z1)
y1=np.imag(z1)
z1al=dife(x1)+1j*dife(y1)
gamma1=op.fixed_point(gammaf,gamma,args=(z1,z1al))	

zc2=dt*calcularW(gamma,z1,z1al)+np.conj(z1)
z2=np.conj(zc2)
x2=np.real(z2)
y2=np.imag(z2)
z2al=dife(x2)+1j*dife(y2)
gamma2=op.fixed_point(gammaf,gamma1,args=(z2,z2al))


zc3=dt*calcularW(gamma,z2,z2al)+np.conj(z2)
z3=np.conj(zc3)
x3=np.real(z3)
y3=np.imag(z3)
z3al=dife(x3)+1j*dife(y3)
gamma3=op.fixed_point(gammaf,gamma2,args=(z3,z3al))



b1=55.0
b2=-59.0
b3=37.0
b4=-9.0
paso=0
t=0.0


zactual=1j*np.zeros(N,dtype='float64')
xactual=np.zeros(N,dtype='float64')
yactual=np.zeros(N,dtype='float64')
zactualal=np.zeros(N,dtype='float64')
gammaActual=np.zeros(N,dtype='float64')
while t<0.29:
	zactualC=np.conj(z3)+(dt/24.0)*((b1*calcularW(gamma3,z3,z3al))+(b2*calcularW(gamma2,z2,z2al))+(b3*calcularW(gamma1,z1,z1al))+(b4*calcularW(gamma,z,zal)))
	zactual=np.conj(zactualC)
	xactual=np.real(zactual)
	yactual=np.imag(zactual)
	zactualal=dife(xactual)+1j*dife(yactual)
	gammaActual=op.fixed_point(gammaf,np.zeros(N,dtype='float64'),args=(zactual,zactualal),xtol=0.01)	
	#gammaActual=punto_fijo_gamma(np.zeros(N,dtype='float64'), zactual,zactualal)
	#print(gammaActual)
	gamma=gamma1
	z=z1
	zal=z1al
	gamma1=gamma2
	z1=z2
	z1al=z2al
	gamma2=gamma3
	z2=z3
	z2al=z3al
	gamma3=gammaActual
	z3=zactual
	z3al=zactualal

	if(paso%100==0):
		nombre="datos_t_"+str(paso)+".txt"
		f=open(nombre,'w')
		f.write(str(t)+'\n')
		for i in range(N):
			f.write(str(xactual[i])+','+str(yactual[i])+'\n')
		f.close()
		#plt.plot(xactual,yactual)	
		#plt.draw()
		#plt.pause(0.1)
	t+=dt
	paso+=1
	print("Paso ", paso, " Tiempo ", t)

	
