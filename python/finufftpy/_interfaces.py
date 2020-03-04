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

# default opts
opts_default = nufft_opts()
default_opts(opts_default)
debug_def=opts_default.debug
spread_debug_def=opts_default.spread_debug
spread_sort_def=opts_default.spread_sort
fftw_def=0
modeord_def=opts_default.modeord
chkbnds_def=opts_default.chkbnds
upsampfac_def=opts_default.upsampfac

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

def set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac):
  opts.debug = debug
  opts.spread_debug = spread_debug;
  opts.spread_sort = spread_sort
  opts.fftw = fftwopts(fftw)
  opts.modeord = modeord
  opts.chkbnds = chkbnds
  opts.upsampfac = upsampfac

## easy interfaces
def nufft1d1(x,c,isign,eps,ms,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  c = _cchk(c)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  if f is None and c.ndim==2:
    _f = np.zeros([ms,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([ms], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==ms*ntransf
  assert _f.shape[0]==ms
  assert _f.ndim==c.ndim
  if c.ndim==2:
    assert _f.shape[1]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(1,1,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,None,None,0,None,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)

def nufft1d2(x,c,isign,eps,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and f array, use f to detect ntransf
  x = _rchk(x)
  f = _cchk(f)
  if f.ndim>1:
    assert f.ndim==2
    ntransf = f.shape[1]
  else:
    ntransf = 1
  
  ms = f.shape[0]
  M = x.size
  if c is None and f.ndim==2:
    _c = np.zeros([M,ntransf], dtype=np.complex128, order='F')
  elif c is None:
    _c = np.zeros([M], dtype=np.complex128, order='F')
  else:
    _c = _cchk(c)

  assert x.size*ntransf==_c.size
  assert x.size==_c.shape[0]
  if f.ndim==2:
    assert f.shape[1]==_c.shape[1]

  # number of input pts
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(2,1,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,None,None,0,None,None,None)

  #excute
  info = execute(plan,_c,f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if c is None:
    return _c
  else:
    _copy(_c, c)

def nufft1d3(x,c,isign,eps,s,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  c = _cchk(c)
  s = _rchk(s)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  assert s.ndim==1
  ms=s.size
  if f is None and c.ndim==2:
    _f = np.zeros([ms,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([ms], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==ms*ntransf
  assert _f.shape[0]==ms
  assert _f.ndim==c.ndim
  if c.ndim==2:
    assert _f.shape[1]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(3,1,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,None,None,ms,s,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)
