# finufftpy module, ie python-user-facing access to (no-data-copy) interfaces
#
# This is where default opts are stated (in arg list, but not docstring).

# todo: pass opts as python double array, neater?
# Note: this JFM code is an extra level of wrapping beyond DFM's style.
# Barnett 10/31/17: changed all type-2 not to have ms,etc as an input but infer
#                   from size of f.

# google-style docstrings for napoleon

import finufftpy_cpp
import numpy as np

## David Stein's functions for checking input and output variables
def _rchk(x):
  """
  Check if array x is of the appropriate type
  (float64, F-contiguous in memory)
  If not, produce a copy
  """
  return np.array(x, dtype=np.float64, order='F', copy=False)
def _cchk(x):
  """
  Check if array x is of the appropriate type
  (complex128, F-contiguous in memory)
  If not, produce a copy
  """
  return np.array(x, dtype=np.complex128, order='F', copy=False)
def _copy(_x, x):
  """
  Copy _x to x, only if the underlying data of _x differs from that of x
  """
  if _x.data != x.data:
    x[:] = _x

## makeplan
def makeplan(tp,n_dims,n_modes,iflag,n_transf,tol,blksize,plan,opts):
  return finufftpy_cpp.makeplan(tp,n_dims,n_modes,iflag,n_transf,tol,blksize,plan,opts)

## setpts
def setpts(plan,M,xj,yj,zj,N,s,t,u):
  global _xj,_yj,_zj,_s,_t,_u
  _xj = _rchk(xj)
  _yj = _rchk(yj)
  _zj = _rchk(zj)
  _s = _rchk(s)
  _t = _rchk(t)
  _u = _rchk(u)
  return finufftpy_cpp.setpts(plan,M,_xj,_yj,_zj,N,_s,_t,_u)

## execute
def execute(plan,weights,result):
  _weights = _cchk(weights)
  _result = _cchk(result)
  info = finufftpy_cpp.execute(plan,_weights,_result)
  _copy(_weights,weights)
  _copy(_result,result)
  return info

## easy interfaces
def nufft1d1(x,c,isign,eps,ms,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  c = _cchk(c)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft1d1_cpp(x,c,isign,eps,ms,_f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_f, f)
  return info

def nufft1d2(x,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft1d3(x,c,isign,eps,s,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  return 0

def nufft2d1(x,y,c,isign,eps,ms,mt,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft2d1many(x,y,c,isign,eps,ms,mt,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft2d2(x,y,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft2d2many(x,y,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft2d3(x,y,c,isign,eps,s,t,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  return 0

def nufft3d1(x,y,z,c,isign,eps,ms,mt,mu,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft3d2(x,y,z,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  return 0

def nufft3d3(x,y,z,c,isign,eps,s,t,u,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  return 0
