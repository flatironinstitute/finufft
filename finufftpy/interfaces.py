# the finufftpy module, ie user-facing access to no-data-copy python interfaces
#
# todo: add docstrings for how to call, and set opts from python, here!

import finufftpy_cpp
import numpy as np

## 1-d

def finufft1d1(xj,cj,iflag,eps,ms,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# fk is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	cj=cj.astype(np.complex128,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft1d1_cpp(xj,cj,iflag,eps,ms,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft1d2(xj,cj,iflag,eps,ms,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# cj is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	fk=fk.astype(np.complex128,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft1d2_cpp(xj,cj,iflag,eps,ms,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft1d3(x,c,iflag,eps,s,f,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# f is the output and must have dtype=np.complex128
	x=x.astype(np.float64,copy=False) #copies only if type changes
	c=c.astype(np.complex128,copy=False) #copies only if type changes
	s=s.astype(np.float64,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft1d3_cpp(x,c,iflag,eps,s,f,debug,spread_debug,spread_sort,fftw,modeord)

## 2-d

def finufft2d1(xj,yj,cj,iflag,eps,ms,mt,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# fk is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	yj=yj.astype(np.float64,copy=False) #copies only if type changes
	cj=cj.astype(np.complex128,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft2d1_cpp(xj,yj,cj,iflag,eps,ms,mt,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft2d2(xj,yj,cj,iflag,eps,ms,mt,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# cj is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	yj=yj.astype(np.float64,copy=False) #copies only if type changes
	fk=fk.astype(np.complex128,copy=False,order='F') #copies only if type changes: so if fk is C-ordered, it will make a transposed copy
	return finufftpy_cpp.finufft2d2_cpp(xj,yj,cj,iflag,eps,ms,mt,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft2d3(x,y,c,iflag,eps,s,t,f,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# f is the output and must have dtype=np.complex128
	x=x.astype(np.float64,copy=False) #copies only if type changes
	y=y.astype(np.float64,copy=False) #copies only if type changes
	c=c.astype(np.complex128,copy=False) #copies only if type changes
	s=s.astype(np.float64,copy=False) #copies only if type changes
	t=t.astype(np.float64,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft2d3_cpp(x,y,c,iflag,eps,s,t,f,debug,spread_debug,spread_sort,fftw,modeord)

## 3-d

def finufft3d1(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# fk is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	yj=yj.astype(np.float64,copy=False) #copies only if type changes
	zj=zj.astype(np.float64,copy=False) #copies only if type changes
	cj=cj.astype(np.complex128,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft3d1_cpp(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft3d2(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# cj is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	yj=yj.astype(np.float64,copy=False) #copies only if type changes
	zj=zj.astype(np.float64,copy=False) #copies only if type changes
	fk=fk.astype(np.complex128,copy=False,order='F') #copies only if type changes: so if fk is not F-ordered, it will make a transposed copy
	return finufftpy_cpp.finufft3d2_cpp(xj,yj,zj,cj,iflag,eps,ms,mt,mu,fk,debug,spread_debug,spread_sort,fftw,modeord)

def finufft3d3(x,y,z,c,iflag,eps,s,t,u,f,debug=0,spread_debug=0,spread_sort=1,fftw=0,modeord=0):
	# f is the output and must have dtype=np.complex128
	x=x.astype(np.float64,copy=False) #copies only if type changes
	y=y.astype(np.float64,copy=False) #copies only if type changes
	z=z.astype(np.float64,copy=False) #copies only if type changes
	c=c.astype(np.complex128,copy=False) #copies only if type changes
	s=s.astype(np.float64,copy=False) #copies only if type changes
	t=t.astype(np.float64,copy=False) #copies only if type changes
	u=u.astype(np.float64,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft3d3_cpp(x,y,z,c,iflag,eps,s,t,u,f,debug,spread_debug,spread_sort,fftw,modeord)
