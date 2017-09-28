import numpy as np
import finufftpy
import math
import time

def compute_error(Xest,Xtrue):
	numer=np.sqrt(np.sum(np.abs((Xest-Xtrue))**2));
	denom=np.sqrt(np.sum(np.abs((Xest+Xtrue)/2))**2);
	if (denom!=0):
		return numer/denom
	else:
		return 0

def print_report(label,elapsed,Xest,Xtrue,npts):
	print(label+':')
	print('    Est. error      %g' % (compute_error(Xest,Xtrue)))
	print('    Elapsed (sec)   %g' % (elapsed))
	print('    nu.pts/sec      %g' % (npts/elapsed))
	print('')

def accuracy_speed_tests(num_nonuniform_points,num_uniform_points,eps):
	nj,nk = int(num_nonuniform_points),int(num_nonuniform_points)
	iflag=1
	num_samples=int(np.minimum(20,num_uniform_points*0.5+1)) #for estimating accuracy

	print('Accuracy and speed tests for %d nonuniform points and eps=%g (error estimates use %d samples per run)' % (num_nonuniform_points,eps,num_samples))

	# for doing the error estimates
	Xest=np.zeros(num_samples,dtype=np.complex128)
	Xtrue=np.zeros(num_samples,dtype=np.complex128)

	###### 1-d
	ms=int(num_uniform_points)

	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms],dtype=np.complex128)
	timer=time.time()
	ret=finufftpy.finufft1d1(xj,cj,iflag,eps,ms,fk)
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
	ret=finufftpy.finufft1d2(xj,cj,iflag,eps,ms,fk)
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
	ret=finufftpy.finufft1d3(x,c,iflag,eps,s,f)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*s[ii]*x))
		Xtrue[ii]=f[ii]
	print_report('finufft1d3',elapsed,Xest,Xtrue,nj+nk)

	###### 2-d
	ms=int(np.ceil(num_uniform_points**(1/2)))
	mt=ms

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt],dtype=np.complex128,order='F')
	timer=time.time()
	ret=finufftpy.finufft2d1(xj,yj,cj,iflag,eps,ms,mt,fk)
	elapsed=time.time()-timer

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(cj * np.exp(1j*(Ks.ravel()[ii]*xj+Kt.ravel()[ii]*yj)))
		Xtrue[ii]=fk.ravel()[ii]
	print_report('finufft2d1',elapsed,Xest,Xtrue,nj)

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt)+1j*np.random.rand(ms,mt);
	timer=time.time()
	ret=finufftpy.finufft2d2(xj,yj,cj,iflag,eps,ms,mt,fk)
	elapsed=time.time()-timer

	Ks,Kt=np.mgrid[-np.floor(ms/2):np.floor((ms-1)/2+1),-np.floor(mt/2):np.floor((mt-1)/2+1)]
	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii])))
		Xtrue[ii]=cj[ii]
	print_report('finufft2d2',elapsed,Xest,Xtrue,nj)

	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	timer=time.time()
	ret=finufftpy.finufft2d3(x,y,c,iflag,eps,s,t,f)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*(s[ii]*x+t[ii]*y)))
		Xtrue[ii]=f[ii]
	print_report('finufft2d3',elapsed,Xest,Xtrue,nj+nk)

	###### 3-d
	ms=int(np.ceil(num_uniform_points**(1/3)))
	mt=ms
	mu=ms

	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt,mu],dtype=np.complex128,order='F')
	timer=time.time()
	ret=finufftpy.finufft3d1(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)
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
	ret=finufftpy.finufft3d2(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)
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
	ret=finufftpy.finufft3d3(x,y,z,c,iflag,eps,s,t,u,f)
	elapsed=time.time()-timer

	for ii in np.arange(0,num_samples):
		Xest[ii]=np.sum(c * np.exp(1j*(s[ii]*x+t[ii]*y+u[ii]*z)))
		Xtrue[ii]=f[ii]
	print_report('finufft3d3',elapsed,Xest,Xtrue,nj+nk)