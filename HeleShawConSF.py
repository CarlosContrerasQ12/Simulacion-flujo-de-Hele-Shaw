import numpy as np
from scipy import fftpack as sp
from scipy import optimize as op
from scipy import integrate as inte
from scipy import interpolate as ip
import matplotlib.pyplot as plt
from scipy import signal as sg
from numba import jit
import os
import time
import warnings
warnings.filterwarnings('ignore')
os.chdir('FinalA08S001/')

N=256
alphas=np.arange(0,2*np.pi,2*np.pi/N)
h=2*np.pi/N
dt=1e-5
Amu=0.8
S=0.01

print("Sera este el fin del hombre araña?")
print("Numero de puntos en la discretizacion: ",np.size(alphas))
print("dt: ",dt)
print("Amu: ",Amu )
print("S: ",S)

sf=np.fft.fftfreq(N)
k=sf*N

@jit
def der(zs):
	dfou=1.0j * k * sp.fft(zs)
	ins=np.where(np.abs(dfou)<1e-12)
	dfou[ins]=0
	return np.real(sp.ifft(dfou))

@jit	
def integrar(zs):
	yi=1j*np.zeros(N)
	fz=sp.fft(zs)
	yi[1:]=fz[1:]/(1j*k[1:])
	ins=np.where(np.abs(yi)<1e-12)
	yi[ins]=0.0
	return np.real(sp.ifft(yi))	
	
	
@jit
def filtrar(z):
	ff=sp.fft(z)
	ins=np.where(np.abs(ff)<1e-12)
	ff[ins]=0.0
	return sp.ifft(ff)
	
@jit
def filtrarReales(z):
	ff=sp.fft(z)
	ins=np.where(np.abs(ff)<1e-12)
	ff[ins]=0.0
	return np.real(np.fft.ifft(ff))
	
@jit
def filtrar_fourier(z):
	ins=np.where(np.abs(z)<1e-12)
	z[ins]=0.0
	return z
	
@jit
def funcion_gamma(g,z,zd,dkappa):
	resp=np.zeros(N)
	for i in range(N):
		integral=0.0
		for j in range(N):
			if abs(j-i)%2==1:
				integral+=(g[j]/(z[i]-z[j]))
		resp[i]=2*Amu*np.real((-zd[i]/z[i])+zd[i]*(-1j/(2*np.pi))*2*h*integral)+S*dkappa[i]
	return resp
	
@jit
def zero_gamma(g,z,zd,dkappa):
	return g-funcion_gamma(g,z,zd,dkappa)
	
@jit
def iteracion_puntofijo_gamma(g0,z,zd,dkappa):
	for k in range(60):
		g0=funcion_gamma(g0,z,zd,dkappa)	
	return g0
	
@jit
def calcularW(z,zd,g):
	wr=0j*np.zeros(N)
	for i in range(N):
		integral=0.0
		for j in range(N):
			if abs(j-i)%2==1:
				integral+=(g[j]/(z[i]-z[j]))
		wr[i]=(-1.0/(z[i]))-((1j/(2*np.pi))*integral*2*h)
	return wr
	
@jit
def integra_cero_alpha(arr):
	return integrar(arr-np.mean(arr))+np.mean(arr)*(alphas)

@jit
def integrar_hasta_fin(arr):
	return inte.trapz(np.append(arr,arr[0]),dx=h)

@jit
def calcular_theta(thetadev):
	theta=integra_cero_alpha(thetadev)+np.pi*0.5
	return theta
	
@jit
def calcular_T(thetadev,u):
	pr=thetadev*u
	primer=integra_cero_alpha(pr)
	primer=primer-primer[0]
	segundo=(alphas/(2*np.pi))*integrar_hasta_fin(pr)
	return primer-segundo
	
@jit
def calcular_P_fourier(L,Ud,T,thetadev,thetaf):
	s=L/(2*np.pi)
	p1=np.fft.fft(Ud)/s
	p2=(0.5*S/(s**3))*(k**3)*np.sign(k)*thetaf
	p3=np.fft.fft(thetadev*T)/s
	return p1+p2+p3
	
@jit
def reconstruirXY(La,theta):
	#xi=(La/(2*np.pi))*(sp.diff(np.cos(theta),order=-1))
	#yi=(La/(2*np.pi))*(sp.diff(np.sin(theta),order=-1))
	xi=(La/(2*np.pi))*(integra_cero_alpha(np.cos(theta)))
	yi=(La/(2*np.pi))*(integra_cero_alpha(np.sin(theta)))
	return xi,(yi-yi[0])-0.1

@jit
def calcular_paso_runge_kutta(theta0d,U0,T0,L0,theta0f):
	k1=-dt*integrar_hasta_fin(theta0d*U0)
	cte0=-(0.5*S*(2*np.pi/L0)**3)*np.sign(k)*k**3
	l1=dt*(cte0*theta0f+calcular_P_fourier(L0,der(U0),T0,theta0d,theta0f))
	
	L1=L0+0.5*k1
	theta1f=theta0f+0.5*l1
	s1=L1/(2*np.pi)
	theta1=np.real(sp.ifft(theta1f))
	x1,y1=reconstruirXY(L1,theta1)
	z1=x1+1j*y1
	x1d=der(x1)
	y1d=der(y1)
	z1d=x1d+1j*y1d
	kappa1=(x1d*der(y1d)-y1d*der(x1d))/(s1*s1*s1)
	theta1d=kappa1*s1
	g1=iteracion_puntofijo_gamma(np.zeros(N),z1,z1d,der(kappa1))
	W1=calcularW(z1,z1d,g1)
	U1=-np.imag(z1d*W1)/s1
	T1=calcular_T(theta1d,U1)
	
	k2=-dt*(integrar_hasta_fin(theta1d*U1))
	cte1=-(0.5*S*(2*np.pi/L1)**3)*np.sign(k)*k**3
	l2=dt*(cte1*theta1f+calcular_P_fourier(L1,der(U1),T1,theta1d,theta1f))
	
	L2=L0+0.5*k2
	theta2f=theta0f+0.5*l2
	s2=L2/(2*np.pi)
	theta2=np.real(sp.ifft(theta2f))
	x2,y2=reconstruirXY(L2,theta2)
	z2=x2+1j*y2
	x2d=der(x2)
	y2d=der(y2)
	z2d=x2d+1j*y2d
	kappa2=(x2d*der(y2d)-y2d*der(x2d))/(s2*s2*s2)
	theta2d=kappa2*s2
	g2=iteracion_puntofijo_gamma(np.zeros(N),z2,z2d,der(kappa2))
	W2=calcularW(z2,z2d,g2)
	U2=-np.imag(z2d*W2)/s2
	T2=calcular_T(theta2d,U2)
	
	k3=-dt*(integrar_hasta_fin(theta2d*U2))
	cte2=-(0.5*S*(2*np.pi/L2)**3)*np.sign(k)*k**3
	l3=dt*(cte2*theta2f+calcular_P_fourier(L2,der(U2),T2,theta2d,theta2f))
	
	L3=L0+k3
	theta3f=theta0f+l3
	s3=L3/(2*np.pi)
	theta3=np.real(sp.ifft(theta3f))
	x3,y3=reconstruirXY(L3,theta3)
	z3=x3+1j*y3
	x3d=der(x3)
	y3d=der(y3)
	z3d=x3d+1j*y3d
	kappa3=(x3d*der(y3d)-y3d*der(x3d))/(s3*s3*s3)
	theta3d=kappa3*s3
	g3=iteracion_puntofijo_gamma(np.zeros(N),z3,z3d,der(kappa3))
	W3=calcularW(z3,z3d,g3)
	U3=-np.imag(z3d*W3)/s3
	T3=calcular_T(theta3d,U3)
	
	k4=-dt*(integrar_hasta_fin(theta3d*U3))
	cte3=-(0.5*S*(2*np.pi/L3)**3)*np.sign(k)*k**3
	l4=dt*(cte3*theta3f+calcular_P_fourier(L3,der(U3),T3,theta3d,theta3f))
	
	return L0+(1.0/6.0)*(k1+2*k2+2*k3+k4),theta0f+(1.0/6.0)*(l1+2*l2+2*l3+l4)	
	
@jit
def extrapolar(g0,g1,g2,g3):
	gi=np.zeros(N)
	for i in range(N):
		f=ip.InterpolatedUnivariateSpline([1,2,3,4], [g0[i],g1[i],g2[i],g3[i]],k=1,ext=0)	
		gi[i]=f(5)
	return gi
	
#Estado inicial
L0=2*np.pi
x0=np.cos(alphas)
y0=np.sin(alphas)-0.1
z0=x0+1j*y0	
x0d=der(x0)
y0d=der(y0)
z0d=x0d+1j*y0d
s0=L0/(2*np.pi)
kappa0=(x0d*der(y0d)-y0d*der(x0d))/(s0*s0*s0)
theta0=calcular_theta(kappa0*s0)
x0,y0=reconstruirXY(L0,theta0)
theta0d=kappa0*s0
theta0f=np.fft.fft(theta0)
g0=iteracion_puntofijo_gamma(np.zeros(N),z0,z0d,der(kappa0))
W0=calcularW(z0,z0d,g0)
U0=-np.imag(z0d*W0)/s0
T0=calcular_T(theta0d,U0)
P0f=calcular_P_fourier(L0,der(U0),T0,theta0d,theta0f)
fL0=-integrar_hasta_fin(theta0d*U0)

#Primer paso

L1,theta1f=calcular_paso_runge_kutta(theta0d,U0,T0,L0,theta0f)
s1=L1/(2*np.pi)
theta1=np.real(sp.ifft(theta1f))
x1,y1=reconstruirXY(L1,theta1)
z1=x1+1j*y1
x1d=der(x1)
y1d=der(y1)
z1d=x1d+1j*y1d
kappa1=(x1d*der(y1d)-y1d*der(x1d))/(s1*s1*s1)
theta1d=kappa1*s1
g1=iteracion_puntofijo_gamma(np.zeros(N),z1,z1d,der(kappa1))
W1=calcularW(z1,z1d,g1)
U1=-np.imag(z1d*W1)/s1
T1=calcular_T(theta1d,U1)
P1f=calcular_P_fourier(L1,der(U1),T1,theta1d,theta1f)
fL1=-integrar_hasta_fin(theta1d*U1)

#Segundo Paso

L2,theta2f=calcular_paso_runge_kutta(theta1d,U1,T1,L1,theta1f)
s2=L2/(2*np.pi)
theta2=np.real(sp.ifft(theta2f))
x2,y2=reconstruirXY(L2,theta2)
z2=x2+1j*y2
x2d=der(x2)
y2d=der(y2)
z2d=x2d+1j*y2d
kappa2=(x2d*der(y2d)-y2d*der(x2d))/(s2*s2*s2)
theta2d=kappa2*s2
g2=iteracion_puntofijo_gamma(np.zeros(N),z2,z2d,der(kappa2))
W2=calcularW(z2,z2d,g2)
U2=-np.imag(z2d*W2)/s2
T2=calcular_T(theta2d,U2)
P2f=calcular_P_fourier(L2,der(U2),T2,theta2d,theta2f)
fL2=-integrar_hasta_fin(theta2d*U2)

#Tercer Paso
L3,theta3f=calcular_paso_runge_kutta(theta2d,U2,T2,L2,theta2f)
s3=L3/(2*np.pi)
theta3=np.real(sp.ifft(theta3f))
x3,y3=reconstruirXY(L3,theta3)
z3=x3+1j*y3
x3d=der(x3)
y3d=der(y3)
z3d=x3d+1j*y3d
kappa3=(x3d*der(y3d)-y3d*der(x3d))/(s3*s3*s3)
theta3d=kappa3*s3
g3=iteracion_puntofijo_gamma(np.zeros(N),z3,z3d,der(kappa3))
W3=calcularW(z3,z3d,g3)
U3=-np.imag(z3d*W3)/s3
T3=calcular_T(theta3d,U3)
P3f=calcular_P_fourier(L3,der(U3),T3,theta3d,theta3f)
fL3=-integrar_hasta_fin(theta3d*U3)



#Definiciones del estado actual
Lacutual=0
sactual=0
thetaActualf=1j*np.zeros(N)
thetaActual=np.zeros(N)
xactual=np.zeros(N)
yactual=np.zeros(N)
xactuald=np.zeros(N)
yactuald=np.zeros(N)
zactual=1j*np.zeros(N)
zactuald=1j*np.zeros(N)
kappaActual=np.zeros(N)
thetaActuald=np.zeros(N)
gActual=np.zeros(N)
Wactual=0j*np.zeros(N)
Uactual=np.zeros(N)
Tactual=np.zeros(N)
Pactualf=1j*np.zeros(N)
fLactual=0


#Comienzo del método
b1=55.0
b2=-59.0
b3=37.0
b4=-9.0
paso=4
t=4*dt

tin=time.time()
plt.grid()
plt.axis('equal')

while(t<0.35):
	t0=time.time()
	Lactual=L3+(dt/24.0)*(b1*fL3+b2*fL2+b3*fL1+b4*fL0)
	sactual=Lactual/(2*np.pi)
	thetaActualf=(4*P3f-6*P2f+4*P1f-P0f+(1.0/dt)*(4*theta3f-3*theta2f+(4.0/3.0)*theta1f-0.25*theta0f))/((25.0/(12.0*dt))+(0.5*S*(1.0/sactual)**3*np.sign(k)*k**3))
	thetaActualf=filtrar_fourier(thetaActualf)
	thetaActual=np.real(sp.ifft(thetaActualf))
	xactual,yactual=reconstruirXY(Lactual,thetaActual)
	xactual=filtrarReales(xactual)
	yactual=filtrarReales(yactual)
	zactual=xactual+1j*yactual
	xactuald=der(xactual)
	yactuald=der(yactual)
	zactuald=xactuald+1j*yactuald
	kappaActual=(xactuald*der(yactuald)-yactuald*der(xactuald))/(sactual*sactual*sactual)
	thetaActuald=kappaActual*sactual
	gi=extrapolar(g0,g1,g2,g3)
	gActual=iteracion_puntofijo_gamma(gi,zactual,zactuald,der(kappaActual))
	Wactual=calcularW(zactual,zactuald,gActual)
	Uactual=-np.imag(zactuald*Wactual)/sactual
	Tactual=calcular_T(thetaActuald,Uactual)
	Pactualf=calcular_P_fourier(Lactual,der(Uactual),Tactual,thetaActuald,thetaActualf)
	fLactual=-integrar_hasta_fin(thetaActuald*Uactual)
	
	
	L0=L1
	s0=s1
	theta0f=theta1f
	theta0=theta1
	x0=x1
	y0=y1
	z0=z1
	z0d=z1d
	kappa0=kappa1
	theta0d=theta1d
	g0=g1
	W0=W1
	U0=U1
	T0=T1
	P0f=P1f
	fL0=fL1
	
	L1=L2
	s1=s2
	theta1f=theta2f
	theta1=theta2
	x1=x2
	y1=y2
	z1=z2
	z1d=z2d
	kappa1=kappa2
	theta1d=theta2d
	g1=g2
	W1=W2
	U1=U2
	T1=T2
	P1f=P2f
	fL1=fL2
	
	L2=L3
	s2=s3
	theta2f=theta3f
	theta2=theta3
	x2=x3
	y2=y3
	z2=z3
	z2d=z3d
	kappa2=kappa3
	theta2d=theta3d
	g2=g3
	W2=W3
	U2=U3
	T2=T3
	P2f=P3f
	fL2=fL3
	
	
	L3=Lactual
	s3=sactual
	theta3f=thetaActualf
	theta3=thetaActual
	x3=xactual
	y3=yactual
	z3=zactual
	z3d=zactuald
	kappa3=kappaActual
	theta3d=thetaActuald
	g3=gActual
	W3=Wactual
	U3=Uactual
	T3=Tactual
	P3f=Pactualf
	fL3=fLactual
	
	t+=dt
	paso+=1
	print("Paso ", paso, " Tiempo ", t, " Lactual ",Lactual, " Calculado en ",time.time()-t0, " Tiempo total ",time.time()-tin)
	
	if(paso%100==0):
		nombre="datos_t_"+str(paso)+".txt"
		f=open(nombre,'w')
		f.write(str(t)+'\n')
		if(xactual[2]!=xactual[2]):
			print("Malditos nan")
			break
		for i in range(N):
			f.write(str(xactual[i])+','+str(yactual[i])+','+str(thetaActual[i])+','+str(kappaActual[i])+'\n')
		f.close()
		plt.plot(xactual,yactual)	
		plt.draw()
		plt.pause(0.0000001)
		
	elif paso>26000 and paso%10==0:
		nombre="datos_t_"+str(paso)+".txt"
		f=open(nombre,'w')
		f.write(str(t)+'\n')
		if(xactual[2]!=xactual[2]):
			print("Malditos nan")
			break
		for i in range(N):
			f.write(str(xactual[i])+','+str(yactual[i])+','+str(thetaActual[i])+','+str(kappaActual[i])+'\n')
		f.close()
		

print("Tiempo total", time.time()-tin)	

