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

from finufftpy_cpp import default_opts
from finufftpy_cpp import destroy
from finufftpy_cpp import nufft_opts
from finufftpy_cpp import finufft_plan
from finufftpy_cpp import fftwopts
from finufftpy_cpp import get_max_threads

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

  assert x.size==c.size
  assert f.size==ms

  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  default_opts(opts)
  opts.debug = debug
  opts.spread_debug = spread_debug;
  opts.spread_sort = spread_sort
  opts.fftw = fftwopts(fftw)
  opts.modeord = modeord
  opts.chkbnds = chkbnds
  opts.upsampfac = upsampfac

  #plan
  plan = finufft_plan()
  info = makeplan(1,1,n_modes,isign,1,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,None,None,0,None,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  #info = finufftpy_cpp.finufft1d1_cpp(x,c,isign,eps,ms,_f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  _copy(_f, f)
  return info

def nufft1d2(x,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  f = _cchk(f)
  _c = _cchk(c)
  info = finufftpy_cpp.finufft1d2_cpp(x,_c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_c, c)
  return info

def nufft1d3(x,c,isign,eps,s,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  x = _rchk(x)
  c = _cchk(c)
  s = _rchk(s)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft1d3_cpp(x,c,isign,eps,s,_f,debug,spread_debug,spread_sort,fftw,upsampfac)
  _copy(_f, f)
  return info

def nufft2d1(x,y,c,isign,eps,ms,mt,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  c = _cchk(c)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft2d1_cpp(x,y,c,isign,eps,ms,mt,_f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_f, f)
  return info

def nufft2d1many(x,y,c,isign,eps,ms,mt,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  c = _cchk(c)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft2d1many_cpp(x,y,c,isign,eps,ms,mt,_f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_f, f)
  return info

def nufft2d2(x,y,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  f = _cchk(f)
  _c = _cchk(c)
  info = finufftpy_cpp.finufft2d2_cpp(x,y,_c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_c, c)
  return info

def nufft2d2many(x,y,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  f = _cchk(f)
  _c = _cchk(c)
  info = finufftpy_cpp.finufft2d2many_cpp(x,y,_c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_c, c)
  return info

def nufft2d3(x,y,c,isign,eps,s,t,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  c = _cchk(c)
  s = _rchk(s)
  t = _rchk(t)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft2d3_cpp(x,y,c,isign,eps,s,t,_f,debug,spread_debug,spread_sort,fftw,upsampfac)
  _copy(_f, f)
  return info

def nufft3d1(x,y,z,c,isign,eps,ms,mt,mu,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  c = _cchk(c)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft3d1_cpp(x,y,z,c,isign,eps,ms,mt,mu,_f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_f, f)
  return info

def nufft3d2(x,y,z,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  f = _cchk(f)
  _c = _cchk(c)
  info = finufftpy_cpp.finufft3d2_cpp(x,y,z,_c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)
  _copy(_c, c)
  return info

def nufft3d3(x,y,z,c,isign,eps,s,t,u,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,upsampfac=2.0):
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  c = _cchk(c)
  s = _rchk(s)
  t = _rchk(t)
  u = _rchk(u)
  _f = _cchk(f)
  info = finufftpy_cpp.finufft3d3_cpp(x,y,z,c,isign,eps,s,t,u,_f,debug,spread_debug,spread_sort,fftw,upsampfac)
  _copy(_f, f)
  return info
