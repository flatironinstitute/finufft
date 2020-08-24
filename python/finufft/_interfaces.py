# finufft module, ie python-user-facing access to (no-data-copy) interfaces
#
# Some default opts are stated here (in arg list, but not docstring).

# Barnett 10/31/17: changed all type-2 not to have ms,etc as an input but infer
#                   from size of f.
# Barnett 2018?: google-style docstrings for napoleon.
# Lu 03/10/20: added guru interface calls
# Anden 8/18/20: auto-made docstrings for the 9 simple/many routines


import finufft._finufft as _finufft
import numpy as np
import warnings
import numbers


from finufft._finufft import default_opts
from finufft._finufft import nufft_opts
from finufft._finufft import finufft_plan
from finufft._finufft import finufftf_plan


### Plan class definition
class Plan:
    def __init__(self,tp,n_modes_or_dim,eps=1e-6,isign=None,n_trans=1,**kwargs):
        # set default iflag based on if iflag is None
        if iflag==None:
            if tp==2:
                iflag = -1
            else:
                iflag = 1

        # set opts and check precision type
        opts = nufft_opts()
        default_opts(opts)
        is_single = setkwopts(opts,**kwargs)

        # construct plan based on precision type and eps default value
        if is_single:
            plan = finufftf_plan()
        else:
            plan = finufft_plan()

        # setting n_modes and dim for makeplan
        n_modes = np.ones([3], dtype=np.int64)
        if tp==3:
            npdim = np.asarray(n_modes_or_dim, dtype=np.int)
            if npdim.size != 1:
                raise RuntimeError('FINUFFT type 3 plan n_modes_or_dim must be one number, the dimension')
            dim = int(npdim)
        else:
            npmodes = np.asarray(n_modes_or_dim, dtype=np.int64)
            if npmodes.size>3 or npmodes.size<1:
                raise RuntimeError("FINUFFT n_modes dimension must be 1, 2, or 3")
            dim = int(npmodes.size)
            n_modes[0:dim] = npmodes[::-1]

        # call makeplan based on precision type
        if is_single:
            ier = _finufft.makeplanf(tp,dim,n_modes,iflag,n_trans,eps,plan,opts)
        else:
            ier = _finufft.makeplan(tp,dim,n_modes,iflag,n_trans,eps,plan,opts)

        # check error
        if ier != 0:
            err_handler(ier)

        # set C++ side plan as inner_plan
        self.inner_plan = plan

        # set properties
        self.type = tp
        self.dim = dim
        self.n_modes = n_modes
        self.n_trans = n_trans


    ### setpts
    def setpts(self,xj=None,yj=None,zj=None,s=None,t=None,u=None):
        is_single = is_single_plan(self.inner_plan)

        if is_single:
            # array sanity check
            self._xjf = _rchkf(xj)
            self._yjf = _rchkf(yj)
            self._zjf = _rchkf(zj)
            self._sf = _rchkf(s)
            self._tf = _rchkf(t)
            self._uf = _rchkf(u)

            # valid sizes
            dim = self.dim
            tp = self.type
            (self.nj, self.nk) = valid_setpts(tp, dim, self._xjf, self._yjf, self._zjf, self._sf, self._tf, self._uf)

            # call set pts for single prec plan
            if self.dim == 1:
                ier = _finufft.setptsf(self.inner_plan,self.nj,self._xjf,self._yjf,self._zjf,self.nk,self._sf,self._tf,self._uf)
            elif self.dim == 2:
                ier = _finufft.setptsf(self.inner_plan,self.nj,self._yjf,self._xjf,self._zjf,self.nk,self._tf,self._sf,self._uf)
            elif self.dim == 3:
                ier = _finufft.setptsf(self.inner_plan,self.nj,self._zjf,self._yjf,self._xjf,self.nk,self._uf,self._tf,self._sf)
            else:
                raise RuntimeError("FINUFFT dimension must be 1, 2, or 3")
        else:
            # array sanity check
            self._xj = _rchk(xj)
            self._yj = _rchk(yj)
            self._zj = _rchk(zj)
            self._s = _rchk(s)
            self._t = _rchk(t)
            self._u = _rchk(u)

            # valid sizes
            dim = self.dim
            tp = self.type
            (self.nj, self.nk) = valid_setpts(tp, dim, self._xj, self._yj, self._zj, self._s, self._t, self._u)

            # call set pts for double prec plan
            if self.dim == 1:
                ier = _finufft.setpts(self.inner_plan,self.nj,self._xj,self._yj,self._zj,self.nk,self._s,self._t,self._u)
            elif self.dim == 2:
                ier = _finufft.setpts(self.inner_plan,self.nj,self._yj,self._xj,self._zj,self.nk,self._t,self._s,self._u)
            elif self.dim == 3:
                ier = _finufft.setpts(self.inner_plan,self.nj,self._zj,self._yj,self._xj,self.nk,self._u,self._t,self._s)
            else:
                raise RuntimeError("FINUFFT dimension must be 1, 2, or 3")

        if ier != 0:
            err_handler(ier)


    ### execute
    def execute(self,data,out=None):
        is_single = is_single_plan(self.inner_plan)

        if is_single:
            _data = _cchkf(data)
            _out = _cchkf(out)
        else:
            _data = _cchk(data)
            _out = _cchk(out)

        tp = self.type
        n_trans = self.n_trans
        nj = self.nj
        nk = self.nk
        dim = self.dim

        if tp==1 or tp==2:
            ms = self.n_modes[0]
            mt = self.n_modes[1]
            mu = self.n_modes[2]

        # input shape and size check
        if tp==2:
            valid_fshape(data.shape,n_trans,dim,ms,mt,mu,None,2)
        else:
            valid_cshape(data.shape,nj,n_trans)

        # out shape and size check
        if out is not None:
            if tp==1:
                valid_fshape(out.shape,n_trans,dim,ms,mt,mu,None,1)
            if tp==2:
                valid_cshape(out.shape,nj,n_trans)
            if tp==3:
                valid_fshape(out.shape,n_trans,dim,None,None,None,nk,3)

        # allocate out if None
        if out is None:
            if is_single:
                pdtype=np.complex64
            else:
                pdtype=np.complex128
            if tp==1:
                _out = np.squeeze(np.zeros([n_trans, mu, mt, ms], dtype=pdtype, order='C'))
            if tp==2:
                _out = np.squeeze(np.zeros([n_trans, nj], dtype=pdtype, order='C'))
            if tp==3:
                _out = np.squeeze(np.zeros([n_trans, nk], dtype=pdtype, order='C'))

        # call execute based on type and precision type
        if tp==1 or tp==3:
            if is_single:
                ier = _finufft.executef(self.inner_plan,_data,_out)
            else:
                ier = _finufft.execute(self.inner_plan,_data,_out)
        elif tp==2:
            if is_single:
                ier = _finufft.executef(self.inner_plan,_out,_data)
            else:
                ier = _finufft.execute(self.inner_plan,_out,_data)
        else:
            ier = 10

        # check error
        if ier != 0:
            err_handler(ier)

        # return out
        if out is None:
            return _out
        else:
            _copy(_out,out)
            return out


    def __del__(self):
        destroy(self.inner_plan)
        self.inner_plan = None
### End of Plan class definition



### David Stein's functions for checking input and output variables
def _rchk(x):
    """
    Check if array x is of the appropriate type
    (float64, C-contiguous in memory)
    If not, produce a copy
    """
    if x is not None and x.dtype is not np.dtype('float64'):
        raise RuntimeError('FINUFFT data type must be float64 for double precision, data may have mixed precision types')
    return np.array(x, dtype=np.float64, order='C', copy=False)
def _cchk(x):
    """
    Check if array x is of the appropriate type
    (complex128, C-contiguous in memory)
    If not, produce a copy
    """
    if x is not None and (x.dtype is not np.dtype('complex128') and x.dtype is not np.dtype('float64')):
        raise RuntimeError('FINUFFT data type must be complex128 for double precision, data may have mixed precision types')
    return np.array(x, dtype=np.complex128, order='C', copy=False)
def _rchkf(x):
    """
    Check if array x is of the appropriate type
    (float64, C-contiguous in memory)
    If not, produce a copy
    """
    if x is not None and x.dtype is not np.dtype('float32'):
        raise RuntimeError('FINUFFT data type must be float32 for single precision, data may have mixed precision types')
    return np.array(x, dtype=np.float32, order='C', copy=False)
def _cchkf(x):
    """
    Check if array x is of the appropriate type
    (complex128, C-contiguous in memory)
    If not, produce a copy
    """
    if x is not None and (x.dtype is not np.dtype('complex64') and x.dtype is not np.dtype('float32')):
        raise RuntimeError('FINUFFT data type must be complex64 for single precision, data may have mixed precision types')
    return np.array(x, dtype=np.complex64, order='C', copy=False)
def _copy(_x, x):
    """
    Copy _x to x, only if the underlying data of _x differs from that of x
    """
    if _x.data != x.data:
        x[:] = _x


### error handler
def err_handler(ier):
    switcher = {
        1: 'FINUFFT eps tolerance too small to achieve',
        2: 'FINUFFT malloc size requested greater than MAXNF',
        3: 'FINUFFT spreader fine grid too small compared to kernel width',
        4: 'FINUFFT spreader nonuniform point out of range [-3pi,3pi]^d in type 1 or 2',
        5: 'FINUFFT spreader malloc error',
        6: 'FINUFFT spreader illegal direction (must be 1 or 2)',
        7: 'FINUFFT opts.upsampfac not > 1.0',
        8: 'FINUFFT opts.upsampfac not a value with known Horner polynomial rule',
        9: 'FINUFFT number of transforms ntrans invalid',
        10: 'FINUFFT transform type invalid',
        11: 'FINUFFT general malloc failure',
        12: 'FINUFFT number of dimensions dim invalid'
    }
    err_msg = switcher.get(ier,'Unknown error')

    if ier == 1:
        warnings.warn(err_msg, Warning)
    else:
        raise RuntimeError(err_msg)


### valid sizes when setpts
def valid_setpts(tp,dim,x,y,z,s,t,u):
    if x.ndim != 1:
        raise RuntimeError('FINUFFT x must be a vector')

    nj = x.size

    if tp == 3:
        nk = s.size
        if s.ndim != 1:
            raise RuntimeError('FINUFFT s must be a vector')
    else:
        nk = 0

    if dim > 1:
        if y.ndim != 1:
            raise RuntimeError('FINUFFT y must be a vector')
        if y.size != nj:
            raise RuntimeError('FINUFFT y must have same length as x')
        if tp==3:
            if t.ndim != 1:
                raise RuntimeError('FINUFFT t must be a vector')
            if t.size != nk:
                raise RuntimeError('FINUFFT t must have same length as s')

    if dim > 2:
        if z.ndim != 1:
            raise RuntimeError('FINUFFT z must be a vector')
        if z.size != nj:
            raise RuntimeError('FINUFFT z must have same length as x')
        if tp==3:
            if u.ndim != 1:
                raise RuntimeError('FINUFFT u must be a vector')
            if u.size != nk:
                raise RuntimeError('FINUFFT u must have same length as s')

    return (nj, nk)


### ntransf for type 1 and type 2
def valid_ntr_tp12(dim,shape,n_transin,n_modesin):
    if len(shape) == dim+1:
        n_trans = shape[0]
        n_modes = shape[1:dim+1]
    elif len(shape) == dim:
        n_trans = 1
        n_modes = shape
    else:
        raise RuntimeError('FINUFFT type 1 output dimension or type 2 input dimension must be either dim or dim+1(n_trans>1)')

    if n_transin is not None and n_trans != n_transin:
        raise RuntimeError('FINUFFT input n_trans and output n_trans do not match')

    if n_modesin is not None:
        if n_modes != n_modesin:
            raise RuntimeError('FINUFFT input n_modes and output n_modes do not match')

    return (n_trans,n_modes)


### valid number of transforms
def valid_ntr(x,c):
    n_trans = int(c.size/x.size)
    if n_trans*x.size != c.size:
        raise RuntimeError('FINUFFT c.size must be divisible by x.size')
    valid_cshape(c.shape,x.size,n_trans)
    return n_trans


### valid shape of c
def valid_cshape(cshape,xsize,n_trans):
    if n_trans == 1:
        if len(cshape) != 1:
            raise RuntimeError('FINUFFT c.ndim must be 1 if n_trans = 1')
        if cshape[0] != xsize:
            raise RuntimeError('FINUFFT c.size must be same as x.size if n_trans = 1')
    if n_trans > 1:
        if len(cshape) != 2:
            raise RuntimeError('FINUFFT c.ndim must be 2 if n_trans > 1')
        if cshape[1] != xsize or cshape[0] != n_trans:
            raise RuntimeError('FINUFFT c.shape must be (n_trans, x.size) if n_trans > 1')


### valid shape of f
def valid_fshape(fshape,n_trans,dim,ms,mt,mu,nk,tp):
    if tp == 3:
        if n_trans == 1:
            if len(fshape) != 1:
                raise RuntimeError('FINUFFT f.ndim must be 1 for type 3 if n_trans = 1')
            if fshape[0] != nk:
                raise RuntimeError('FINUFFT f.size of must be nk if n_trans = 1')
        if n_trans > 1:
            if len(fshape) != 2:
                raise RuntimeError('FINUFFT f.ndim must be 2 for type 3 if n_trans > 1')
            if fshape[1] != nk or fshape[0] != n_trans:
                raise RuntimeError('FINUFFT f.shape must be (n_trans, nk) if n_trans > 1')
    else:
        if n_trans == 1:
            if len(fshape) != dim:
                raise RuntimeError('FINUFFT f.ndim must be same as the problem dimension for type 1 or 2 if n_trans = 1')
        if n_trans > 1:
            if len(fshape) != dim+1:
                raise RuntimeError('FINUFFT f.ndim must be same as the problem dimension + 1 for type 1 or 2 if n_trans > 1')
            if fshape[0] != n_trans:
                raise RuntimeError('FINUFFT f.shape[0] must be n_trans for type 1 or 2 if n_trans > 1')
        if fshape[-1] != ms:
            raise RuntimeError('FINUFFT f.shape is not consistent with n_modes')
        if dim>1:
            if fshape[-2] != mt:
                raise RuntimeError('FINUFFT f.shape is not consistent with n_modes')
        if dim>2:
            if fshape[-3] != mu:
                raise RuntimeError('FINUFFT f.shape is not consistent with n_modes')


### check if it's a single precision plan
def is_single_plan(plan):
    if type(plan) is _finufft.finufftf_plan:
        return True
    elif type(plan) is _finufft.finufft_plan:
        return False
    else:
        raise RuntimeError('FINUFFT invalid plan type')


### check if dtype is single or double
def is_single_dtype(dtype):
    dtype = np.dtype(dtype)

    if dtype == np.dtype('float64') or dtype == np.dtype('complex128'):
        return False
    elif dtype == np.dtype('float32') or dtype == np.dtype('complex64'):
        return True
    else:
        raise RuntimeError('FINUFFT dtype(precision type) must be single or double')


### kwargs opt set
def setkwopts(opt,**kwargs):
    warnings.simplefilter('always')

    dtype = 'double'
    for key,value in kwargs.items():
        if hasattr(opt,key):
            setattr(opt,key,value)
        elif key == 'dtype':
            dtype = value
        else:
            warnings.warn('Warning: nufft_opts does not have attribute "' + key + '"', Warning)

    warnings.simplefilter('default')

    return is_single_dtype(dtype)


### destroy
def destroy(plan):
    if is_single_plan(plan):
        ier = _finufft.destroyf(plan)
    else:
        ier = _finufft.destroy(plan)

    if ier != 0:
        err_handler(ier)


### invoke guru interface, this function is used for simple interfaces
def invoke_guru(dim,tp,x,y,z,c,s,t,u,f,isign,eps,n_modes,**kwargs):
    # infer dtype from x
    if x.dtype is np.dtype('float64'):
        pdtype = 'double'
    elif x.dtype is np.dtype('float32'):
        pdtype = 'single'
    else:
        raise RuntimeError('FINUFFT x dtype should be float64 for double precision or float32 for single precision')
    # check n_modes type, n_modes must be a tuple or an integer
    if n_modes is not None:
        if (not isinstance(n_modes, tuple)) and (not isinstance(n_modes, numbers.Integral)):
            raise RuntimeError('FINUFFT input n_modes must be a tuple or an integer')
    # sanity check for n_modes input as tuple
    if isinstance(n_modes, tuple):
        if len(n_modes) != dim:
            raise RuntimeError('FINUFFT input n_modes dimension does not match problem dimension')
        if (not all(isinstance(elmi, numbers.Integral) for elmi in n_modes)):
            raise RuntimeError('FINUFFT all elements of input n_modes must be integer')
    # if n_modes is an integer populate n_modes for all dimensions
    if isinstance(n_modes, numbers.Integral):
        n_modes = (n_modes,)*dim

    # infer n_modes/n_trans from input/output
    if tp==1:
        n_trans = valid_ntr(x,c)
        if n_modes is None and f is None:
            raise RuntimeError('FINUFFT type 1 input must supply n_modes or output vector, or both')
        if f is not None:
            (n_trans,n_modes) = valid_ntr_tp12(dim,f.shape,n_trans,n_modes)
    elif tp==2:
        (n_trans,n_modes) = valid_ntr_tp12(dim,f.shape,None,None)
    else:
        n_trans = valid_ntr(x,c)

    #plan
    if tp==3:
        plan = Plan(tp,dim,eps,isign,n_trans,**dict(kwargs,dtype=pdtype))
    else:
        plan = Plan(tp,n_modes,eps,isign,n_trans,**dict(kwargs,dtype=pdtype))

    #setpts
    plan.setpts(x,y,z,s,t,u)

    #excute
    if tp==1 or tp==3:
        out = plan.execute(c,f)
    else:
        out = plan.execute(f,c)

    return out


def _set_nufft_doc(f, dim, tp, example='python/test/accuracy_speed_tests.py'):
    doc_nufft1 = \
    """{dim}D type-1 (nonuniform to uniform) complex NUFFT

    ::

      {pt_spacing}        M-1
      f[{pt_idx}] = SUM c[j] exp(+/-i {pt_inner})
      {pt_spacing}        j=0

          for {pt_constraint}

    Args:
{pts_doc}
      c         (complex[M] or complex[ntransf, M]): source strengths.
      n_modes   (integer or integer tuple of length {dim}, optional): number of
                uniform Fourier modes requested {modes_tuple}. May be even or odd; in
                either case, modes {pt_idx} are integers satisfying
                {pt_constraint}. Must be specified if ``out`` is not given.
      out       (complex[{modes}] or complex[ntransf, {modes}], optional): output array
                for Fourier mode values. If ``n_modes`` is specifed, the shape
                must match, otherwise ``n_modes`` is inferred from ``out``.
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      eps       (float, optional): precision requested (>1e-16).
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[{modes}] or complex[ntransf, {modes}]: The resulting array.

    Example:
      See ``{example}``.
    """

    doc_nufft2 = \
    """{dim}D type-2 (uniform to nonuniform) complex NUFFT

    ::

      c[j] = SUM f[{pt_idx}] exp(+/-i {pt_inner})       for j = 0, ..., M-1
             {pt_idx}

          where the sum is over {pt_constraint}

    Args:
{pts_doc}
      f         (complex[{modes}] or complex[ntransf, {modes}]): Fourier mode
                coefficients, where {modes} may be even or odd. In either case
                the mode indices {pt_idx} satisfy {pt_constraint}.
      out       (complex[M] or complex[ntransf, M], optional): output array
                at targets.
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      eps       (float, optional): precision requested (>1e-16).
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[M] or complex[ntransf, M]: The resulting array.

    Example:
      See ``{example}``.
    """

    doc_nufft3 = \
    """{dim}D type-3 (nonuniform to nonuniform) complex NUFFT

    ::

               M-1
      f[k]  =  SUM   c[j] exp(+/-i {pt_inner_type3}),      for k = 0, ..., N-1
               j=0


    Args:
{src_pts_doc}
      c         (complex[M] or complex[ntransf, M]): source strengths.
{target_pts_doc}
      out       (complex[N] or complex[ntransf, N]): output values at target frequencies.
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      eps       (float, optional): precision requested (>1e-16).
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[M] or complex[ntransf, M]: The resulting array.

    Example:
      See ``{example}``.
    """

    doc_nufft = {1: doc_nufft1, 2: doc_nufft2, 3: doc_nufft3}

    pts = ('x', 'y', 'z')
    target_pts = ('s', 't', 'u')

    dims = range(1, dim + 1)

    v = {}

    v['dim'] = dim
    v['example'] = example

    v['modes'] = ', '.join('N{}'.format(i) for i in dims)
    v['modes_tuple'] = '(' + v['modes'] + (', ' if dim == 1 else '') + ')'
    v['pt_idx'] = ', '.join('k{}'.format(i) for i in dims)
    v['pt_spacing'] = ' ' * (len(v['pt_idx']) - 2)
    v['pt_inner'] = ' + '.join('k{0} {1}(j)'.format(i, x) for i, x in zip(dims, pts[:dim]))
    v['pt_constraint'] = ', '.join('-N{0}/2 <= k{0} <= (N{0}-1)/2'.format(i) for i in dims)
    v['pts_doc'] = '\n'.join('      {}         (float[M]): nonuniform points, valid only in [-3pi, 3pi].'.format(x) for x in pts[:dim])

    # for type 3 only
    v['src_pts_doc'] = '\n'.join('      {}         (float[M]): nonuniform source points.'.format(x) for x in pts[:dim])
    v['target_pts_doc'] = '\n'.join('      {}         (float[N]): nonuniform target points.'.format(x) for x in target_pts[:dim])
    v['pt_inner_type3'] = ' + '.join('{0}[k] {1}[j]'.format(s, x) for s, x in zip(target_pts[:dim], pts[:dim]))

    if dim > 1:
        v['pt_inner'] = '(' + v['pt_inner'] + ')'
        v['pt_inner_type3'] = '(' + v['pt_inner_type3'] + ')'

    f.__doc__ = doc_nufft[tp].format(**v)


### easy interfaces
### 1d1
def nufft1d1(x,c,n_modes=None,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(1,1,x,None,None,c,None,None,None,out,isign,eps,n_modes,**kwargs)


### 1d2
def nufft1d2(x,f,out=None,eps=1e-6,isign=-1,**kwargs):
    return invoke_guru(1,2,x,None,None,out,None,None,None,f,isign,eps,None,**kwargs)


### 1d3
def nufft1d3(x,c,s,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(1,3,x,None,None,c,s,None,None,out,isign,eps,None,**kwargs)


### 2d1
def nufft2d1(x,y,c,n_modes=None,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(2,1,x,y,None,c,None,None,None,out,isign,eps,n_modes,**kwargs)


### 2d2
def nufft2d2(x,y,f,out=None,eps=1e-6,isign=-1,**kwargs):
    return invoke_guru(2,2,x,y,None,out,None,None,None,f,isign,eps,None,**kwargs)


### 2d3
def nufft2d3(x,y,c,s,t,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(2,3,x,y,None,c,s,t,None,out,isign,eps,None,**kwargs)


### 3d1
def nufft3d1(x,y,z,c,n_modes=None,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(3,1,x,y,z,c,None,None,None,out,isign,eps,n_modes,**kwargs)


### 3d2
def nufft3d2(x,y,z,f,out=None,eps=1e-6,isign=-1,**kwargs):
    return invoke_guru(3,2,x,y,z,out,None,None,None,f,isign,eps,None,**kwargs)


### 3d3
def nufft3d3(x,y,z,c,s,t,u,out=None,eps=1e-6,isign=1,**kwargs):
    return invoke_guru(3,3,x,y,z,c,s,t,u,out,isign,eps,None,**kwargs)


_set_nufft_doc(nufft1d1, 1, 1, 'python/examples/simple1d1.py, python/examples/simpleopts1d1.py')
_set_nufft_doc(nufft1d2, 1, 2)
_set_nufft_doc(nufft1d3, 1, 3)
_set_nufft_doc(nufft2d1, 2, 1, 'python/examples/simple2d1.py, python/examples/many2d1.py')
_set_nufft_doc(nufft2d2, 2, 2)
_set_nufft_doc(nufft2d3, 2, 3)
_set_nufft_doc(nufft3d1, 3, 1)
_set_nufft_doc(nufft3d2, 3, 2)
_set_nufft_doc(nufft3d3, 3, 3)