import numpy as np
import finufftpy
import math
import time

def speed_tests(num_nonuniform_pts,eps):

	nj,nk = int(num_nonuniform_pts),int(num_nonuniform_pts)
	ms,mt,mu = 200,200,200
	iflag=1

	print('Using %g nonuniform points and eps=%g' % (num_nonuniform_pts,eps))

	###### 1-d

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms],dtype=np.complex128)
	ret=finufftpy.finufft1d1(xj,cj,iflag,eps,ms,fk)

	elapsed=time.time() - timer
	str='finufft1d1'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms)+1j*np.random.rand(ms);
	ret=finufftpy.finufft1d2(xj,cj,iflag,eps,ms,fk)

	elapsed=time.time() - timer
	str='finufft1d2'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	x=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft1d3(x,c,iflag,eps,s,f)

	elapsed=time.time() - timer
	str='finufft1d3'
	nups=(nj+nk)/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	###### 2-d

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt],dtype=np.complex128,order='F')
	ret=finufftpy.finufft2d1(xj,yj,cj,iflag,eps,ms,mt,fk)

	elapsed=time.time() - timer
	str='finufft2d1'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt)+1j*np.random.rand(ms,mt);
	ret=finufftpy.finufft2d2(xj,yj,cj,iflag,eps,ms,mt,fk)

	elapsed=time.time() - timer
	str='finufft2d2'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft2d3(x,y,c,iflag,eps,s,t,f)

	elapsed=time.time() - timer
	str='finufft2d3'
	nups=(nj+nk)/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	###### 3-d

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.random.rand(nj)+1j*np.random.rand(nj);
	fk=np.zeros([ms,mt,mu],dtype=np.complex128,order='F')
	ret=finufftpy.finufft3d1(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)

	elapsed=time.time() - timer
	str='finufft3d1'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	xj=np.random.rand(nj)*2*math.pi-math.pi
	yj=np.random.rand(nj)*2*math.pi-math.pi
	zj=np.random.rand(nj)*2*math.pi-math.pi
	cj=np.zeros([nj],dtype=np.complex128);
	fk=np.random.rand(ms,mt,mu)+1j*np.random.rand(ms,mt,mu);
	ret=finufftpy.finufft3d2(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk)

	elapsed=time.time() - timer
	str='finufft3d2'
	nups=nj/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))

	timer = time.time()
	x=np.random.rand(nj)*2*math.pi-math.pi
	y=np.random.rand(nj)*2*math.pi-math.pi
	z=np.random.rand(nj)*2*math.pi-math.pi
	c=np.random.rand(nj)+1j*np.random.rand(nj);
	s=np.random.rand(nk)*2*math.pi-math.pi
	t=np.random.rand(nk)*2*math.pi-math.pi
	u=np.random.rand(nk)*2*math.pi-math.pi
	f=np.zeros([nk],dtype=np.complex128)
	ret=finufftpy.finufft3d3(x,y,z,c,iflag,eps,s,t,u,f)

	elapsed=time.time() - timer
	str='finufft3d3'
	nups=(nj+nk)/elapsed;
	print('Elapsed time for %s (sec): %g (%g n.u.pts/sec)' % (str,elapsed,nups))