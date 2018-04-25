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

## 1-d

def nufft1d1(x,c,isign,eps,ms,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """1D type-1 (aka adjoint) complex nonuniform fast Fourier transform

  ::

    	     nj-1
    f(k1) =  SUM c[j] exp(+/-i k1 x(j))  for -ms/2 <= k1 <= (ms-1)/2
             j=0

  Args:
    x     (float[nj]): nonuniform source points, valid only in [-3pi,3pi]
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    ms    (int): number of Fourier modes requested, may be even or odd;
        in either case the modes are integers lying in [-ms/2, (ms-1)/2]
    f     (complex[ms]): output Fourier mode values. Should be initialized as a
              numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown).
    spread_debug (int, optional): 0 (silent), 1, 2... (prints spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (do sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)
	
  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python_tests/demo1d1.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False)        # copies only if type changes
  c=c.astype(np.complex128,copy=False)     # "
  return finufftpy_cpp.finufft1d1_cpp(x,c,isign,eps,ms,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft1d2(x,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """1D type-2 (aka forward) complex nonuniform fast Fourier transform

  ::

    c[j] = SUM   f[k1] exp(+/-i k1 x[j])      for j = 0,...,nj-1
            k1 

	where sum is over -ms/2 <= k1 <= (ms-1)/2.

  Args:
    x     (float[nj]): nonuniform target points, valid only in [-3pi,3pi]
    c     (complex[nj]): output values at targets. Should be initialized as a
        numpy array of the correct size
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    f     (complex[ms]): Fourier mode coefficients, where ms is even or odd
          In either case the mode indices are integers in [-ms/2, (ms-1)/2]
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)

  .. note::

    The output is written into the c array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # c is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  f=f.astype(np.complex128,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft1d2_cpp(x,c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft1d3(x,c,isign,eps,s,f,debug=0,spread_debug=0,spread_sort=2,fftw=0):
  """1D type-3 (NU-to-NU) complex nonuniform fast Fourier transform

  ::

	     nj-1
    f[k]  =  SUM   c[j] exp(+-i s[k] x[j]),      for k = 0, ..., nk-1
	     j=0
  
  Args:
    x     (float[nj]): nonuniform source points, in R
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    s     (float[nk]): nonuniform target frequency points, in R
    f     (complex[nk]): output values at target frequencies.
       Should be initialized as a numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)

  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  c=c.astype(np.complex128,copy=False) #copies only if type changes
  s=s.astype(np.float64,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft1d3_cpp(x,c,isign,eps,s,f,debug,spread_debug,spread_sort,fftw)

## 2-d

def nufft2d1(x,y,c,isign,eps,ms,mt,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """2D type-1 (aka adjoint) complex nonuniform fast Fourier transform

  ::

	            nj-1
	f(k1,k2) =  SUM c[j] exp(+/-i (k1 x(j) + k2 y[j])),
	            j=0
	                  for -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2

  Args:
    x     (float[nj]): nonuniform source x-coords, valid only in [-3pi,3pi]
    y     (float[nj]): nonuniform source y-coords, valid only in [-3pi,3pi]
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    ms    (int): number of Fourier modes in x-direction, may be even or odd;
        in either case the modes are integers lying in [-ms/2, (ms-1)/2]
    mt    (int): number of Fourier modes in y-direction, may be even or odd;
        in either case the modes are integers lying in [-mt/2, (mt-1)/2]

    f     (complex[ms,mt]): output Fourier mode values. Should be initialized as a Fortran-ordered (ie ms fast, mt slow) numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (prints spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)
	
  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python/tests/accuracy_speed_tests.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  c=c.astype(np.complex128,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft2d1_cpp(x,y,c,isign,eps,ms,mt,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft2d2(x,y,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """2D type-2 (aka forward) complex nonuniform fast Fourier transform

  ::

    c[j] =   SUM   f[k1,k2] exp(+/-i (k1 x[j] + k2 y[j])),  for j = 0,...,nj-1
	    k1,k2

    where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2

  Args:
    x     (float[nj]): nonuniform target x-coords, valid only in [-3pi,3pi]
    y     (float[nj]): nonuniform target y-coords, valid only in [-3pi,3pi]
    c     (complex[nj]): output values at targets. Should be initialized as a
        numpy array of the correct size
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    f     (complex[ms,mt]): Fourier mode coefficients, where ms and mt are
          either even or odd; in either case
	  their mode range is integers lying in [-m/2, (m-1)/2], with
	  mode ordering in all dimensions given by modeord.  Ordering is Fortran-style, ie ms fastest.
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)

  .. note::

    The output is written into the c array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # c is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  f=f.astype(np.complex128,copy=False,order='F') #copies only if type changes: so if f is C-ordered, it will make a transposed copy
  return finufftpy_cpp.finufft2d2_cpp(x,y,c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft2d3(x,y,c,isign,eps,s,t,f,debug=0,spread_debug=0,spread_sort=2,fftw=0):
  """2D type-3 (NU-to-NU) complex nonuniform fast Fourier transform

  ::

             nj-1
    f[k]  =  SUM   c[j] exp(+-i s[k] x[j] + t[k] y[j]),  for k = 0,...,nk-1
             j=0
 
  Args:
    x     (float[nj]): nonuniform source point x-coords, in R
    y     (float[nj]): nonuniform source point y-coords, in R
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    s     (float[nk]): nonuniform target x-frequencies, in R
    t     (float[nk]): nonuniform target y-frequencies, in R
    f     (complex[nk]): output values at target frequencies.
       Should be initialized as a numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)

  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  c=c.astype(np.complex128,copy=False) #copies only if type changes
  s=s.astype(np.float64,copy=False) #copies only if type changes
  t=t.astype(np.float64,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft2d3_cpp(x,y,c,isign,eps,s,t,f,debug,spread_debug,spread_sort,fftw)

## 3-d

def nufft3d1(x,y,z,c,isign,eps,ms,mt,mu,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """3D type-1 (aka adjoint) complex nonuniform fast Fourier transform

  ::

	           nj-1
    f(k1,k2,k3) =  SUM c[j] exp(+/-i (k1 x(j) + k2 y[j] + k3 z[j])),
	           j=0
       for -ms/2 <= k1 <= (ms-1)/2,
	   -mt/2 <= k2 <= (mt-1)/2,  -mu/2 <= k3 <= (mu-1)/2

  Args:
    x     (float[nj]): nonuniform source x-coords, valid only in [-3pi,3pi]
    y     (float[nj]): nonuniform source y-coords, valid only in [-3pi,3pi]
    z     (float[nj]): nonuniform source z-coords, valid only in [-3pi,3pi]
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    ms    (int): number of Fourier modes in x-direction, may be even or odd;
        in either case the modes are integers lying in [-ms/2, (ms-1)/2]
    mt    (int): number of Fourier modes in y-direction, may be even or odd;
        in either case the modes are integers lying in [-mt/2, (mt-1)/2]
    mu    (int): number of Fourier modes in z-direction, may be even or odd;
        in either case the modes are integers lying in [-mu/2, (mu-1)/2]

    f     (complex[ms,mt,mu]): output Fourier mode values. Should be initialized as a Fortran-ordered (ie ms fastest) numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (prints spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)
	
  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  z=z.astype(np.float64,copy=False) #copies only if type changes
  c=c.astype(np.complex128,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft3d1_cpp(x,y,z,c,isign,eps,ms,mt,mu,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft3d2(x,y,z,c,isign,eps,f,debug=0,spread_debug=0,spread_sort=2,fftw=0,modeord=0,chkbnds=1):
  """3D type-2 (aka forward) complex nonuniform fast Fourier transform

  ::

    c[j] =   SUM   f[k1,k2,k3] exp(+/-i (k1 x[j] + k2 y[j] + k3 z[j])).
	   k1,k2,k3
	             for j = 0,...,nj-1,  where sum is over
    -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2, -mu/2 <= k3 <= (mu-1)/2

  Args:
    x     (float[nj]): nonuniform target x-coords, valid only in [-3pi,3pi]
    y     (float[nj]): nonuniform target y-coords, valid only in [-3pi,3pi]
    z     (float[nj]): nonuniform target z-coords, valid only in [-3pi,3pi]
    c     (complex[nj]): output values at targets. Should be initialized as a
        numpy array of the correct size
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    f     (complex[ms,mt,mu]): Fourier mode coefficients, where ms, mt and mu
          are either even or odd; in either case
	  their mode range is integers lying in [-m/2, (m-1)/2], with
	  mode ordering in all dimensions given by modeord. Ordering is Fortran-style, ie ms fastest.
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)
    modeord (int, optional): 0 (CMCL increasing mode ordering), 1 (FFT ordering)
    chkbnds (int, optional): 0 (don't check NU points valid), 1 (do)

  .. note::

    The output is written into the c array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF,
       4 at least one NU point out of range (if chkbnds true)

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # c is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  z=z.astype(np.float64,copy=False) #copies only if type changes
  f=f.astype(np.complex128,copy=False,order='F') #copies only if type changes: so if f is not F-ordered, it will make a transposed copy
  return finufftpy_cpp.finufft3d2_cpp(x,y,z,c,isign,eps,f,debug,spread_debug,spread_sort,fftw,modeord,chkbnds)

def nufft3d3(x,y,z,c,isign,eps,s,t,u,f,debug=0,spread_debug=0,spread_sort=2,fftw=0):
  """3D type-3 (NU-to-NU) complex nonuniform fast Fourier transform

  ::

             nj-1
    f[k]  =  SUM   c[j] exp(+-i s[k] x[j] + t[k] y[j] + u[k] z[j]),
             j=0
	                                               for k = 0,...,nk-1
 
  Args:
    x     (float[nj]): nonuniform source point x-coords, in R
    y     (float[nj]): nonuniform source point y-coords, in R
    z     (float[nj]): nonuniform source point z-coords, in R
    c     (complex[nj]): source strengths
    isign (int): if >=0, uses + sign in exponential, otherwise - sign
    eps   (float): precision requested (>1e-16)
    s     (float[nk]): nonuniform target x-frequencies, in R
    t     (float[nk]): nonuniform target y-frequencies, in R
    u     (float[nk]): nonuniform target z-frequencies, in R
    f     (complex[nk]): output values at target frequencies.
       Should be initialized as a numpy array of the correct size
    debug (int, optional): 0 (silent), 1 (print timing breakdown)
    spread_debug (int, optional): 0 (silent), 1, 2... (print spreader info)
    spread_sort (int, optional): 0 (don't sort NU pts in spreader), 1 (sort),
       2 (heuristic decision to sort)
    fftw (int, optional): 0 (use FFTW_ESTIMATE), 1 (use FFTW_MEASURE)

  .. note::

    The output is written into the f array.

  Returns:
    int: 0 if success, 1 if eps too small,
       2 if size of arrays to malloc exceed MAX_NF

  Example:
    see ``python_tests/accuracy_speed_tests.py``
  """
  # f is the output and must have dtype=np.complex128
  x=x.astype(np.float64,copy=False) #copies only if type changes
  y=y.astype(np.float64,copy=False) #copies only if type changes
  z=z.astype(np.float64,copy=False) #copies only if type changes
  c=c.astype(np.complex128,copy=False) #copies only if type changes
  s=s.astype(np.float64,copy=False) #copies only if type changes
  t=t.astype(np.float64,copy=False) #copies only if type changes
  u=u.astype(np.float64,copy=False) #copies only if type changes
  return finufftpy_cpp.finufft3d3_cpp(x,y,z,c,isign,eps,s,t,u,f,debug,spread_debug,spread_sort,fftw)
