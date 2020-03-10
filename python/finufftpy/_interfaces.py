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
## 1D
def nufft1d1(x,c,isign,eps,ms,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  c = _cchk(c)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert c.shape[0]==x.size
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
  assert x.size==c.shape[0]
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  assert s.ndim==1
  nk=s.size
  if f is None and c.ndim==2:
    _f = np.zeros([nk,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([nk], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==nk*ntransf
  assert _f.shape[0]==nk
  assert _f.ndim==c.ndim
  if c.ndim==2:
    assert _f.shape[1]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = nk
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(3,1,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,None,None,nk,s,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)

## 2D
def nufft2d1(x,y,c,isign,eps,ms,mt,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  c = _cchk(c)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert c.shape[0]==x.size
  assert x.size*ntransf==c.size
  assert y.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  if f is None and c.ndim==2:
    _f = np.zeros([ms,mt,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([ms,mt], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==ms*mt*ntransf
  assert _f.shape[0]==ms
  assert _f.shape[1]==mt
  assert _f.ndim==(c.ndim+1)
  if c.ndim==2:
    assert _f.shape[2]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  n_modes[1] = mt
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(1,2,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,None,0,None,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)

def nufft2d2(x,y,c,isign,eps,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and f array, use f to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  f = _cchk(f)
  if f.ndim>2:
    assert f.ndim==3
    ntransf = f.shape[2]
  else:
    assert f.ndim==2
    ntransf = 1
  
  ms = f.shape[0]
  mt = f.shape[1]
  M = x.size
  if c is None and f.ndim==3:
    _c = np.zeros([M,ntransf], dtype=np.complex128, order='F')
  elif c is None:
    _c = np.zeros([M], dtype=np.complex128, order='F')
  else:
    _c = _cchk(c)

  assert x.size*ntransf==_c.size
  assert y.size*ntransf==_c.size
  assert x.size==_c.shape[0]
  if f.ndim==3:
    assert f.shape[2]==_c.shape[1]

  # number of input pts
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  n_modes[1] = mt
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(2,2,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,None,0,None,None,None)

  #excute
  info = execute(plan,_c,f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if c is None:
    return _c
  else:
    _copy(_c, c)

def nufft2d3(x,y,c,isign,eps,s,t,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  c = _cchk(c)
  s = _rchk(s)
  t = _rchk(t)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert x.size==c.shape[0]
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  assert s.ndim==1
  assert t.ndim==1
  assert s.size==t.size
  nk=s.size
  if f is None and c.ndim==2:
    _f = np.zeros([nk,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([nk], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==nk*ntransf
  assert _f.shape[0]==nk
  assert _f.ndim==c.ndim
  if c.ndim==2:
    assert _f.shape[1]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = nk
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(3,2,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,None,nk,s,t,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)

## 3D
def nufft3d1(x,y,z,c,isign,eps,ms,mt,mu,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  c = _cchk(c)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert x.size==c.shape[0]
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  if f is None and c.ndim==2:
    _f = np.zeros([ms,mt,mu,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([ms,mt,mu], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==ms*mt*mu*ntransf
  assert _f.shape[0]==ms
  assert _f.shape[1]==mt
  assert _f.shape[2]==mu
  assert _f.ndim==(c.ndim+2)
  if c.ndim==2:
    assert _f.shape[3]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  n_modes[1] = mt
  n_modes[2] = mu
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(1,3,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,z,0,None,None,None)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)

def nufft3d2(x,y,z,c,isign,eps,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and f array, use f to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  f = _cchk(f)
  if f.ndim>3:
    assert f.ndim==4
    ntransf = f.shape[3]
  else:
    ntransf = 1
  
  ms = f.shape[0]
  mt = f.shape[1]
  mu = f.shape[2]
  M = x.size
  if c is None and f.ndim==4:
    _c = np.zeros([M,ntransf], dtype=np.complex128, order='F')
  elif c is None:
    _c = np.zeros([M], dtype=np.complex128, order='F')
  else:
    _c = _cchk(c)

  assert x.size*ntransf==_c.size
  assert x.size==_c.shape[0]
  if f.ndim==4:
    assert f.shape[3]==_c.shape[1]

  # number of input pts
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = ms
  n_modes[1] = mt
  n_modes[2] = mu
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(2,3,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,z,0,None,None,None)

  #excute
  info = execute(plan,_c,f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if c is None:
    return _c
  else:
    _copy(_c, c)

def nufft3d3(x,y,z,c,isign,eps,s,t,u,f,debug=debug_def,spread_debug=spread_debug_def,spread_sort=spread_sort_def,fftw=fftw_def,modeord=modeord_def,chkbnds=chkbnds_def,upsampfac=upsampfac_def):
  # check input x and c array, use c to detect ntransf
  x = _rchk(x)
  y = _rchk(y)
  z = _rchk(z)
  c = _cchk(c)
  s = _rchk(s)
  t = _rchk(t)
  u = _rchk(u)
  if c.ndim>1:
    assert c.ndim==2
    ntransf = c.shape[1]
  else:
    ntransf = 1
  assert x.size==c.shape[0]
  assert x.size*ntransf==c.size
  
  # check output f, if none _f is returned as output,
  # otherwise output is stored in f
  assert s.ndim==1
  assert t.ndim==1
  assert u.ndim==1
  assert s.size==t.size
  assert t.size==u.size
  nk=s.size
  if f is None and c.ndim==2:
    _f = np.zeros([nk,ntransf], dtype=np.complex128, order='F')
  elif f is None:
    _f = np.zeros([nk], dtype=np.complex128, order='F')
  else:
    _f = _cchk(f)
  assert _f.size==nk*ntransf
  assert _f.shape[0]==nk
  assert _f.ndim==c.ndim
  if c.ndim==2:
    assert _f.shape[1]==c.shape[1]

  # number of input pts
  M = x.size
  n_modes = np.ones([3], dtype=np.int64)
  n_modes[0] = nk
  blksize = get_max_threads()

  #opts
  opts = nufft_opts()
  set_opts(opts,debug,spread_debug,spread_sort,fftw,modeord,chkbnds,upsampfac)

  #plan
  plan = finufft_plan()
  info = makeplan(3,3,n_modes,isign,ntransf,eps,blksize,plan,opts)

  #setpts
  info = setpts(plan,M,x,y,z,nk,s,t,u)

  #excute
  info = execute(plan,c,_f)

  #destroy
  info = destroy(plan)

  # return _f if f is none else output is in f
  if f is None:
    return _f
  else:
    _copy(_f, f)
