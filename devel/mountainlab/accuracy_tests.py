import numpy as np
import finufftpy
import math

def accuracy_tests(num_nonuniform_points,eps):
	nj,nk = int(num_nonuniform_points),int(num_nonuniform_points)
	ms,mt,mu = 20,40,60
	iflag=1

	###### 1-d

	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms],dtype=np.complex128)
	ret=finufftpy.finufft1d1(xj,cj,iflag,eps,ms,fk)

	k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
	fk0=np.sum(cj * np.exp(1j*k[0]*xj))
	print('Err for finufft1d1: ',np.abs(fk0-fk[0]))

	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms)+1j*np.random.rand(ms);
	ret=finufftpy.finufft1d2(xj,cj,iflag,eps,ms,fk)

	k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
	cj0=np.sum(fk * np.exp(1j*k*xj[0]))
	print('Err for finufft1d2: ',np.abs(cj0-cj[0]))

	x=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft1d3(x,c,iflag,eps,s,f)

	f0=np.sum(c * np.exp(1j*s[0]*x))
	print('Err for finufft1d3: ',np.abs(f0-f[0]))

	###### 2-d

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt],dtype=np.complex128,order='F')
	ret=finufftpy.finufft2d1(xj,yj,cj,iflag,eps,ms,mt,fk)

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
	fk00=np.sum(cj * np.exp(1j*(Ks[0][0]*xj+Kt[0][0]*yj)))
	print('Err for finufft2d1: ',np.abs(fk00-fk[0][0]))

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt)+1j*np.random.rand(ms,mt);
	ret=finufftpy.finufft2d2(xj,yj,cj,iflag,eps,ms,mt,fk)

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
	cj0=np.sum(fk * np.exp(1j*(Ks*xj[0]+Kt*yj[0])))
	print('Err for finufft2d2: ',np.abs(cj0-cj[0]))

	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft2d3(x,y,c,iflag,eps,s,t,f)

	f0=np.sum(c * np.exp(1j*(s[0]*x+t[0]*y)))
	print('Err for finufft2d3: ',np.abs(f0-f[0]))

	###### 3-d

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt,mu],dtype=np.complex128,order='F')
	ret=finufftpy.finufft3d1(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)

	Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
	fk000=np.sum(cj * np.exp(1j*(Ks[0][0][0]*xj+Kt[0][0][0]*yj+Ku[0][0][0]*zj)))
	print('Err for finufft3d1: ',np.abs(fk000-fk[0][0][0]))

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt,mu)+1j*np.random.rand(ms,mt,mu);
	ret=finufftpy.finufft3d2(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)

	Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
	cj0=np.sum(fk * np.exp(1j*(Ks*xj[0]+Kt*yj[0]+Ku*zj[0])))
	print('Err for finufft3d2: ',np.abs(cj0-cj[0]))

	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	z=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	u=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft3d3(x,y,z,c,iflag,eps,s,t,u,f)

	f0=np.sum(c * np.exp(1j*(s[0]*x+t[0]*y+u[0]*z)))
	print('Err for finufft3d3: ',np.abs(f0-f[0]))