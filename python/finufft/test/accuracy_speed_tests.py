# Jeremy Magland, Sept 2017.
# Alex Barnett fixed integer division issue in python v2 vs v3
#              (affected 1/2, 1/3), 10/13/17.
# Removed ms etc from ?d2 interfaces, 10/31/17. Less reruns 2/14/18.
# 2d1many and 2d2many added, painfully, Barnett 7/29/18

import numpy as np
import finufft
import math
import time

def compute_error(Xest,Xtrue):
	numer=np.sqrt(np.sum(np.abs((Xest-Xtrue))**2));
	denom=np.sqrt(np.sum(np.abs(Xtrue)**2));    # rel l2 norm
	if (denom!=0):
		return numer/denom
	else:
		return 0

def print_report(label,elapsed,Xest,Xtrue,npts):
	print(label+':')
	print('    Est rel l2 err  %.3g' % (compute_error(Xest,Xtrue)))
	print('    CPU time (sec)  %.3g' % (elapsed))
	if (elapsed>0):
		print('    tot NU pts/sec  %.3g' % (npts/elapsed))
	print('')

def accuracy_speed_tests(num_nonuniform_points,num_uniform_points,eps):
	nj,nk = int(num_nonuniform_points),int(num_nonuniform_points)
	iflag=1
	num_samples=int(np.minimum(5,num_uniform_points*0.5+1)) # number of outputs used for estimating accuracy; is small for speed

	print('Accuracy and speed tests for %d nonuniform points and eps=%g (error estimates use %d samples per run)' % (num_nonuniform_points,eps,num_samples))

	# for doing the error estimates
	Xest=np.zeros(num_samples,dtype=np.complex128)
	Xtrue=np.zeros(num_samples,dtype=np.complex128)

	###### 1-d cases ........................................................
	ms=int(num_uniform_points)

	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft1d1(xj,cj,ms,fk,eps,iflag)
	elapsed=time.time()-timer

	k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(cj * np.exp(1j*k[ii]*xj))
		Xtrue[ii]=fk[ii]
	print_report('finufft1d1',elapsed,Xest,Xtrue,nj)

	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms)+1j*np.random.rand(ms);
	timer=time.time()
	ret=finufft.nufft1d2(xj,fk,cj,eps,iflag)
	elapsed=time.time()-timer

	k=np.arange(-np.floor(ms/2),np.floor((ms-1)/2+1))
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(fk * np.exp(1j*k*xj[ii]))
		Xtrue[ii]=cj[ii]
	print_report('finufft1d2',elapsed,Xest,Xtrue,nj)

	x=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft1d3(x,c,s,f,eps,iflag)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*s[ii]*x))
		Xtrue[ii]=f[ii]
	print_report('finufft1d3',elapsed,Xest,Xtrue,nj+nk)

	###### 2-d cases ....................................................
	ms=int(np.ceil(np.sqrt(num_uniform_points)))
	mt=ms

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj)
	fk=np.zeros([ms,mt],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft2d1(xj,yj,cj,(ms,mt),fk,eps,iflag)
	elapsed=time.time()-timer

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(cj * np.exp(1j*(Ks.ravel()[ii]*xj+Kt.ravel()[ii]*yj)))
		Xtrue[ii]=fk.ravel()[ii]
	print_report('finufft2d1',elapsed,Xest,Xtrue,nj)

	## 2d1many:
	ndata = 8       # how many vectors to do
	cj=np.array(np.random.rand(ndata,nj)+1j*np.random.rand(ndata,nj))
	fk=np.zeros([ndata,ms,mt],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft2d1(xj,yj,cj,ms,fk,eps,iflag)
	elapsed=time.time()-timer

	dtest = ndata-1    # which of the ndata to test (in 0,..,ndata-1)
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(cj[dtest,:] * np.exp(1j*(Ks.ravel()[ii]*xj+Kt.ravel()[ii]*yj)))   # note fortran-ravel-order needed throughout - mess.
		Xtrue[ii]=fk.ravel()[ii + dtest*ms*mt]       # hack the offset in fk array - has to be better way
	print_report('finufft2d1many',elapsed,Xest,Xtrue,ndata*nj)

	# 2d2
	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt)+1j*np.random.rand(ms,mt);
	timer=time.time()
	ret=finufft.nufft2d2(xj,yj,fk,cj,eps,iflag)
	elapsed=time.time()-timer

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii])))
		Xtrue[ii]=cj[ii]
	print_report('finufft2d2',elapsed,Xest,Xtrue,nj)

	# 2d2many (using same ndata and dtest as 2d1many; see above)
	cj=np.zeros([ndata,nj],dtype=np.complex128);
	fk=np.array(np.random.rand(ndata,ms,mt)+1j*np.random.rand(ndata,ms,mt))
	timer=time.time()
	ret=finufft.nufft2d2(xj,yj,fk,cj,eps,iflag)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(fk[dtest,:,:] * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii])))
		Xtrue[ii]=cj[dtest,ii]
	print_report('finufft2d2many',elapsed,Xest,Xtrue,ndata*nj)
	
	# 2d3
	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft2d3(x,y,c,s,t,f,eps,iflag)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*(s[ii]*x+t[ii]*y)))
		Xtrue[ii]=f[ii]
	print_report('finufft2d3',elapsed,Xest,Xtrue,nj+nk)

	###### 3-d cases ............................................................
	ms=int(np.ceil(num_uniform_points**(1.0/3)))
	mt=ms
	mu=ms

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt,mu],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft3d1(xj,yj,zj,cj,fk.shape,fk,eps,iflag)
	elapsed=time.time()-timer

	Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(cj * np.exp(1j*(Ks.ravel()[ii]*xj+Kt.ravel()[ii]*yj+Ku.ravel()[ii]*zj)))
		Xtrue[ii]=fk.ravel()[ii]
	print_report('finufft3d1',elapsed,Xest,Xtrue,nj)

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt,mu)+1j*np.random.rand(ms,mt,mu);
	timer=time.time()
	ret=finufft.nufft3d2(xj,yj,zj,fk,cj,eps,iflag)
	elapsed=time.time()-timer

	Ks,Kt,Ku=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1),-np.floor(mu/2):np.floor((mu-1)/2+1)]
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii]+Ku*zj[ii])))
		Xtrue[ii]=cj[ii]
	print_report('finufft3d2',elapsed,Xest,Xtrue,nj)

	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	z=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	u=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	timer=time.time()
	ret=finufft.nufft3d3(x,y,z,c,s,t,u,f,eps,iflag)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*(s[ii]*x+t[ii]*y+u[ii]*z)))
		Xtrue[ii]=f[ii]
	print_report('finufft3d3',elapsed,Xest,Xtrue,nj+nk)
