# finufft module, ie python-user-facing access to (no-data-copy) interfaces
#
# Some default opts are stated here (in arg list, but not docstring).

# Barnett 10/31/17: changed all type-2 not to have ms,etc as an input but infer
#                   from size of f.
# Barnett 2018?: google-style docstrings for napoleon.
# Lu 03/10/20: added guru interface calls
# Anden 8/18/20: auto-made docstrings for the 9 simple/many routines


import numpy as np
import warnings
import numbers

from ctypes import byref
from ctypes import c_longlong
from ctypes import c_void_p

import finufft._finufft as _finufft

### Plan class definition
class Plan:
    r"""
    A non-uniform fast Fourier transform (NUFFT) plan

    The ``Plan`` class lets the user exercise more fine-grained control over
    the execution of an NUFFT. First, the plan is created with a certain set
    of parameters (type, mode configuration, tolerance, sign, number of
    simultaneous transforms, and so on). Then the nonuniform points are set
    (source or target depending on the type). Finally, the plan is executed on
    some data, yielding the desired output.

    In the simple interface, all these steps are executed in a single call to
    the ``nufft*`` functions. The benefit of separating plan creation from
    execution is that it allows for plan reuse when certain parameters (like
    mode configuration) or nonuniform points remain the same between different
    NUFFT calls. This becomes especially important for small inputs, where
    execution time may be dominated by initialization steps such as allocating
    and FFTW plan and sorting the nonuniform points.

    Example:
    ::

        import numpy as np
        import finufft

        # set up parameters
        n_modes = (1000, 2000)
        n_pts = 100000
        nufft_type = 1
        n_trans = 4

        # generate nonuniform points
        x = 2 * np.pi * np.random.uniform(size=n_pts)
        y = 2 * np.pi * np.random.uniform(size=n_pts)

        # generate source strengths
        c = (np.random.standard_normal(size=(n_trans, n_pts))
             + 1J * np.random.standard_normal(size=(n_trans, n_pts)))

        # initialize the plan
        plan = finufft.Plan(nufft_type, n_modes, n_trans)

        # set the nonuniform points
        plan.setpts(x, y)

        # execute the plan
        f = plan.execute(c)

    Also see ``python/finufft/examples/guru1d1.py`` and
    ``python/finufft/examples/guru2d1.py``.

    Args:
        nufft_type      (int): type of NUFFT (1, 2, or 3).
        n_modes_or_dim  (int or tuple of ints): for type 1 and type 2, this
                        should be a tuple specifying the number of modes in
                        each dimension (for example, ``(50, 100)``),
                        otherwise, for type 3, this should be the
                        number of dimensions (between 1 and 3).
        n_trans         (int, optional): number of transforms to compute
                        simultaneously.
        eps             (float, optional): precision requested (>1e-16).
        isign           (int, optional): if +1, uses the positive sign
                        exponential, otherwise the negative sign exponential;
                        defaults to +1 for types 1 and 3 and to -1 for type 2.
        dtype           (string, optional): the precision of the transform,
                        ``'complex64'`` or ``'complex128'``.
        **kwargs        (optional): for more options, see :ref:`opts`.
    """
    def __init__(self,nufft_type,n_modes_or_dim,n_trans=1,eps=1e-6,isign=None,dtype='complex128',**kwargs):
        # set default isign based on if isign is None
        if isign==None:
            if nufft_type==2:
                isign = -1
            else:
                isign = 1

        # set opts and check precision type
        opts = _finufft.FinufftOpts()
        _finufft._default_opts(opts)
        setkwopts(opts,**kwargs)

        dtype = np.dtype(dtype)

        if dtype == np.float64:
            warnings.warn("Real dtypes are currently deprecated and will be "
                          "removed in version 2.3. Converting to complex128.",
                          DeprecationWarning)
            dtype = np.complex128

        if dtype == np.float32:
            warnings.warn("Real dtypes are currently deprecated and will be "
                          "removed in version 2.3. Converting to complex64.",
                          DeprecationWarning)
            dtype = np.complex64

        is_single = is_single_dtype(dtype)

        # construct plan based on precision type and eps default value
        plan = c_void_p(None)

        # setting n_modes and dim for makeplan
        if nufft_type==3:
            npdim = np.asarray(n_modes_or_dim, dtype=np.int64)
            if npdim.size != 1:
                raise RuntimeError('FINUFFT type 3 plan n_modes_or_dim must be one number, the dimension')
            dim = int(npdim)
            n_modes = np.ones([dim], dtype=np.int64)
        else:
            npmodes = np.asarray(n_modes_or_dim, dtype=np.int64)
            if npmodes.size>3 or npmodes.size<1:
                raise RuntimeError("FINUFFT n_modes dimension must be 1, 2, or 3")
            dim = int(npmodes.size)
            n_modes = np.ones([dim], dtype=np.int64)
            n_modes[0:dim] = npmodes[::-1]

        n_modes = (c_longlong * dim)(*n_modes)

        if is_single:
            self._makeplan = _finufft._makeplanf
            self._setpts = _finufft._setptsf
            self._execute = _finufft._executef
            self._destroy = _finufft._destroyf
        else:
            self._makeplan = _finufft._makeplan
            self._setpts = _finufft._setpts
            self._execute = _finufft._execute
            self._destroy = _finufft._destroy

        ier = self._makeplan(nufft_type, dim, n_modes, isign, n_trans, eps,
                             byref(plan), opts)

        # check error
        if ier != 0:
            err_handler(ier)

        # set C++ side plan as inner_plan
        self.inner_plan = plan

        # set properties
        self.type = nufft_type
        self.dim = dim
        self.n_modes = n_modes
        self.n_trans = n_trans

        if is_single:
            self.dtype = np.dtype("complex64")
        else:
            self.dtype = np.dtype("complex128")


    ### setpts
    def setpts(self,x=None,y=None,z=None,s=None,t=None,u=None):
        r"""
        Set the nonuniform points

        For type 1, this sets the coordinates of the ``M`` nonuniform source
        points, for type 2, it sets the coordinates of the ``M`` target
        points, and for type 3 it sets both the ``M`` source points and the
        ``N`` target points.

        The dimension of the plan determines the number of arguments supplied.
        For example, if ``dim == 2``, we provide ``x`` and ``y`` (as well as
        ``s`` and ``t`` for a type-3 transform).

        Args:
            x       (float[M]): first coordinate of the nonuniform points
                    (source for type 1 and 3, target for type 2).
            y       (float[M], optional): second coordinate of the nonuniform
                    points (source for type 1 and 3, target for type 2).
            z       (float[M], optional): third coordinate of the nonuniform
                    points (source for type 1 and 3, target for type 2).
            s       (float[N], optional): first coordinate of the nonuniform
                    points (target for type 3).
            t       (float[N], optional): second coordinate of the nonuniform
                    points (target for type 3).
            u       (float[N], optional): third coordinate of the nonuniform
                    points (target for type 3).
        """

        real_dtype = _get_real_dtype(self.dtype)

        self._xj = _ensure_array_type(x, "x", real_dtype)
        self._yj = _ensure_array_type(y, "y", real_dtype)
        self._zj = _ensure_array_type(z, "z", real_dtype)
        self._s = _ensure_array_type(s, "s", real_dtype)
        self._t = _ensure_array_type(t, "t", real_dtype)
        self._u = _ensure_array_type(u, "u", real_dtype)

        # valid sizes
        dim = self.dim
        tp = self.type
        (self.nj, self.nk) = valid_setpts(tp, dim, self._xj, self._yj, self._zj, self._s, self._t, self._u)

        # call set pts for single prec plan
        if self.dim == 1:
            ier = self._setpts(self.inner_plan, self.nj, self._xj, self._yj, self._zj, self.nk, self._s, self._t, self._u)
        elif self.dim == 2:
            ier = self._setpts(self.inner_plan, self.nj, self._yj, self._xj, self._zj, self.nk, self._t, self._s, self._u)
        elif self.dim == 3:
            ier = self._setpts(self.inner_plan, self.nj, self._zj, self._yj, self._xj, self.nk, self._u, self._t, self._s)

        if ier != 0:
            err_handler(ier)


    ### execute
    def execute(self,data,out=None):
        r"""
        Execute the plan

        Performs the NUFFT specified at plan instantiation with the points set
        by ``setpts``. For type-1 and type-3 transforms, the input is a set of
        source strengths, while for a type-2 transform, it consists of an
        array of size ``n_modes``. If ``n_trans`` is greater than one,
        ``n_trans`` inputs are expected, stacked along the first axis.

        Args:
            data    (complex[M], complex[n_tr, M], complex[n_modes], or complex[n_tr, n_modes]): The input source strengths
                    (type 1 and 3) or source modes (type 2).
            out     (complex[n_modes], complex[n_tr, n_modes], complex[M], or complex[n_tr, M], optional): The array where the
                    output is stored. Must be of the right size.

        Returns:
            complex[n_modes], complex[n_tr, n_modes], complex[M], or complex[n_tr, M]: The output array of the transform(s).
        """

        _data = _ensure_array_type(data, "data", self.dtype)
        _out = _ensure_array_type(out, "out", self.dtype, output=True)

        tp = self.type
        n_trans = self.n_trans
        nj = self.nj
        nk = self.nk
        dim = self.dim

        if tp==1 or tp==2:
            ms, mt, mu = [*self.n_modes, *([1]*(3-len(self.n_modes)))]

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
            if tp==1:
                _out = np.zeros([*data.shape[:-1], *self.n_modes[::-1]], dtype=self.dtype, order='C')
            if tp==2:
                _out = np.zeros([*data.shape[:-dim], nj], dtype=self.dtype, order='C')
            if tp==3:
                _out = np.zeros([*data.shape[:-1], nk], dtype=self.dtype, order='C')

        # call execute based on type and precision type
        if tp==1 or tp==3:
            ier = self._execute(self.inner_plan,
                                _data.ctypes.data_as(c_void_p),
                                _out.ctypes.data_as(c_void_p))
        elif tp==2:
            ier = self._execute(self.inner_plan,
                                _out.ctypes.data_as(c_void_p),
                                _data.ctypes.data_as(c_void_p))

        # check error
        if ier != 0:
            err_handler(ier)

        return _out


    def __del__(self):
        destroy(self)
        self.inner_plan = None
### End of Plan class definition


def _get_real_dtype(dtype):
    return np.array(0, dtype=dtype).real.dtype


def _ensure_array_type(x, name, dtype, output=False):
    if x is None:
        return np.array(0, dtype=dtype, order="C")

    if x.dtype != dtype:
        raise TypeError(f"Argument `{name}` does not have the correct dtype: {x.dtype} was given, but {dtype} was expected.")

    if not output:
        reqs = ["C"]
    else:
        reqs = ["C", "W"]

    for prop in reqs:
        if not x.flags[prop]:
            if output:
                raise TypeError(f"Argument `{name}` does not satisfy the following requirement: {prop}")
            else:
                warnings.warn(f"Argument `{name}` does not satisfy the following requirement: {prop}. Copying array (this may reduce performance)")
                x = np.array(x, dtype=dtype, order="C")

    return x


### error handler (keep up to date with FINUFFT/include/defs.h)
def err_handler(ier):
    switcher = {
        1: 'FINUFFT eps tolerance too small to achieve',
        2: 'FINUFFT malloc size requested greater than MAX_NF',
        3: 'FINUFFT spreader fine grid too small compared to kernel width',
        4: 'FINUFFT spreader nonuniform point out of range [-pi, pi)^d [DEPRECATED]', # DEPRECATED
        5: 'FINUFFT spreader malloc error',
        6: 'FINUFFT spreader illegal direction (must be 1 or 2)',
        7: 'FINUFFT opts.upsampfac not > 1.0',
        8: 'FINUFFT opts.upsampfac not a value with known Horner polynomial rule',
        9: 'FINUFFT number of transforms ntrans invalid',
        10: 'FINUFFT transform type invalid',
        11: 'FINUFFT general malloc failure',
        12: 'FINUFFT number of dimensions dim invalid',
        13: 'FINUFFT spread_thread option invalid',
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
        raise RuntimeError('FINUFFT type 1 output dimension or type 2 input dimension must be either dim (n_trans==1) or dim+1 (n_trans>=1)')

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
        if len(cshape) != 1 and len(cshape) != 2:
            raise RuntimeError('FINUFFT c.ndim must be 1 or 2 if n_trans == 1')
        if cshape[-1] != xsize or np.prod(cshape) != xsize:
            raise RuntimeError('FINUFFT c.size must be same as x.size if n_trans == 1')
    if n_trans > 1:
        if len(cshape) != 2:
            raise RuntimeError('FINUFFT c.ndim must be 2 if n_trans > 1')
        if cshape[1] != xsize or cshape[0] != n_trans:
            raise RuntimeError('FINUFFT c.shape must be (n_trans, x.size) if n_trans > 1')


### valid shape of f
def valid_fshape(fshape,n_trans,dim,ms,mt,mu,nk,tp):
    if tp == 3:
        if n_trans == 1:
            if len(fshape) != 1 and len(fshape) != 2:
                raise RuntimeError('FINUFFT f.ndim must be 1 or 2 for type 3 if n_trans == 1')
            if fshape[-1] != nk or np.prod(fshape) != nk:
                raise RuntimeError('FINUFFT f.size of must be nk if n_trans == 1')
        if n_trans > 1:
            if len(fshape) != 2:
                raise RuntimeError('FINUFFT f.ndim must be 2 for type 3 if n_trans > 1')
            if fshape[1] != nk or fshape[0] != n_trans:
                raise RuntimeError('FINUFFT f.shape must be (n_trans, nk) if n_trans > 1')
    else:
        if n_trans == 1:
            if len(fshape) != dim and len(fshape) != dim+1:
                raise RuntimeError('FINUFFT f.ndim must be same as the problem dimension or the problem dimension + 1 for type 1 or 2 if n_trans == 1')
            if len(fshape) == dim+1 and fshape[0] != n_trans:
                raise RuntimeError('FINUFFT f.shape[0] must be 1 for type 1 or 2 if n_trans == 1 and len(f.shape) == dim+1')
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


### check if dtype is single or double
def is_single_dtype(dtype):
    dtype = np.dtype(dtype)

    if dtype == np.dtype('complex128'):
        return False
    elif dtype == np.dtype('complex64'):
        return True
    else:
        raise RuntimeError('FINUFFT dtype(precision type) must be single or double')


### kwargs opt set
def setkwopts(opt,**kwargs):

    # Use context manager to mutate `warnings` filter stack
    # This will restore the state of the `warnings` stack on exit from the context.
    with warnings.catch_warnings():
        warnings.simplefilter('always')

        for key,value in kwargs.items():
            if hasattr(opt,key):
                setattr(opt,key,value)
            else:
                warnings.warn('Warning: finufft_opts does not have attribute "' + key + '"', Warning)


### destroy
def destroy(plan):
    if hasattr(plan, "inner_plan"):
        ier = plan._destroy(plan.inner_plan)

        if ier != 0:
            err_handler(ier)


### invoke guru interface, this function is used for simple interfaces
def invoke_guru(dim,tp,x,y,z,c,s,t,u,f,isign,eps,n_modes,**kwargs):
    # infer dtype from x
    if x.dtype is np.dtype('float64'):
        pdtype = 'complex128'
    elif x.dtype is np.dtype('float32'):
        pdtype = 'complex64'
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
        plan = Plan(tp,dim,n_trans,eps,isign,pdtype,**kwargs)
    else:
        plan = Plan(tp,n_modes,n_trans,eps,isign,pdtype,**kwargs)

    #setpts
    plan.setpts(x,y,z,s,t,u)

    #excute
    if tp==1 or tp==3:
        out = plan.execute(c,f)
    else:
        out = plan.execute(f,c)

    return out


def _wrap_docstring(docstring, tw=80, min_spacing=2):
    lines = docstring.expandtabs().splitlines()

    for k, line in enumerate(lines):
        if len(line) > tw:
            last_space = line[:tw].rfind(' ')
            indent_level = line.rfind(' ' * min_spacing) + min_spacing

            lines[k] = line[:last_space]

            new_line = (' ' * indent_level) + line[last_space + 1:]

            # Check if the indentation level continues on next line. If so,
            # concatenate, otherwise insert new line.
            if len(lines[k + 1]) - len(lines[k + 1].lstrip()) >= indent_level:
                lines[k + 1] = new_line + ' ' + lines[k + 1].lstrip()
            else:
                lines.insert(k + 1, new_line)

    docstring = '\n'.join(lines)

    return docstring


def _set_nufft_doc(f, dim, tp, example='python/finufft/test/accuracy_speed_tests.py'):
    doc_nufft1 = \
    """{dim}D type-1 (nonuniform to uniform) complex NUFFT

    ::

      {pt_spacing}        M-1
      f[{pt_idx}] = SUM c[j] exp(+/-i {pt_inner})
      {pt_spacing}        j=0

          for {pt_constraint}

    Args:
{pts_doc}
      c         (complex[M] or complex[n_tr, M]): source strengths.
      n_modes   (integer or integer tuple of length {dim}, optional): number of
                uniform Fourier modes requested {modes_tuple}. May be even or odd; in
                either case, modes {pt_idx} are integers satisfying {pt_constraint}.
                Must be specified if ``out`` is not given.
      out       (complex[{modes}] or complex[n_tr, {modes}], optional): output array
                for Fourier mode values. If ``n_modes`` is specifed, the shape
                must match, otherwise ``n_modes`` is inferred from ``out``.
      eps       (float, optional): precision requested (>1e-16).
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[{modes}] or complex[n_tr, {modes}]: The resulting array.

    Example:
    ::

      import numpy as np
      import finufft

      # number of nonuniform points
      M = 100

      # the nonuniform points
{pts_generate}

      # their complex strengths
      c = (np.random.standard_normal(size=M)
           + 1J * np.random.standard_normal(size=M))

      # desired number of Fourier modes
      {modes} = {sample_modes}

      # calculate the type-1 NUFFT
      f = finufft.nufft{dim}d1({pts}, c, {modes_tuple})

    See also ``{example}``.
    """

    doc_nufft2 = \
    """{dim}D type-2 (uniform to nonuniform) complex NUFFT

    ::

      c[j] = SUM f[{pt_idx}] exp(+/-i {pt_inner})
             {pt_idx}

          for j = 0, ..., M-1, where the sum is over {pt_constraint}

    Args:
{pts_doc}
      f         (complex[{modes}] or complex[n_tr, {modes}]): Fourier mode
                coefficients, where {modes} may be even or odd. In either case
                the mode indices {pt_idx} satisfy {pt_constraint}.
      out       (complex[M] or complex[n_tr, M], optional): output array
                at targets.
      eps       (float, optional): precision requested (>1e-16).
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[M] or complex[n_tr, M]: The resulting array.

    Example:
    ::

      import numpy as np
      import finufft

      # number of nonuniform points
      M = 100

      # the nonuniform points
{pts_generate}

      # number of Fourier modes
      {modes} = {sample_modes}

      # the Fourier mode coefficients
      f = (np.random.standard_normal(size={modes_tuple})
           + 1J * np.random.standard_normal(size={modes_tuple}))

      # calculate the type-2 NUFFT
      c = finufft.nufft{dim}d2({pts}, f)


    See also ``{example}``.
    """

    doc_nufft3 = \
    """{dim}D type-3 (nonuniform to nonuniform) complex NUFFT

    ::

             M-1
      f[k] = SUM c[j] exp(+/-i {pt_inner_type3}),
             j=0

          for k = 0, ..., N-1

    Args:
{src_pts_doc}
      c         (complex[M] or complex[n_tr, M]): source strengths.
{target_pts_doc}
      out       (complex[N] or complex[n_tr, N]): output values at target frequencies.
      eps       (float, optional): precision requested (>1e-16).
      isign     (int, optional): if non-negative, uses positive sign in
                exponential, otherwise negative sign.
      **kwargs  (optional): for more options, see :ref:`opts`.

    .. note::

      The output is written into the ``out`` array if supplied.

    Returns:
      complex[M] or complex[n_tr, M]: The resulting array.

    Example:
    ::

      import numpy as np
      import finufft

      # number of source points
      M = 100

      # number of target points
      N = 200

      # the source points
{pts_generate}

      # the target points
{target_pts_generate}

      # their complex strengths
      c = (np.random.standard_normal(size=M)
           + 1J * np.random.standard_normal(size=M))

      # calcuate the type-3 NUFFT
      f = finufft.nufft{dim}d3({pts}, c, {target_pts})

    See also ``{example}``.
    """

    doc_nufft = {1: doc_nufft1, 2: doc_nufft2, 3: doc_nufft3}

    pts = ('x', 'y', 'z')
    target_pts = ('s', 't', 'u')
    sample_modes = (50, 75, 100)

    dims = range(1, dim + 1)

    v = {}

    v['dim'] = dim

    v['modes'] = ', '.join('N{}'.format(i) for i in dims)
    v['modes_tuple'] = '(' + v['modes'] + (', ' if dim == 1 else '') + ')'
    v['pt_idx'] = ', '.join('k{}'.format(i) for i in dims)
    v['pt_spacing'] = ' ' * (len(v['pt_idx']) - 2)
    v['pt_inner'] = ' + '.join('k{0} {1}(j)'.format(i, x) for i, x in zip(dims, pts[:dim]))
    v['pt_constraint'] = ', '.join('-N{0}/2 <= k{0} <= (N{0}-1)/2'.format(i) for i in dims)
    v['pts_doc'] = '\n'.join('{}(float[M]): nonuniform points, in [-pi, pi), values outside will be folded'.format(x) for x in pts[:dim])

    # for example
    v['pts'] = ', '.join(str(x) for x in pts[:dim])
    v['pts_generate'] = '\n'.join('{} = 2 * np.pi * np.random.uniform(size=M)'.format(x) for x in pts[:dim])
    v['sample_modes'] = ', '.join(str(n) for n in sample_modes[:dim])
    v['example'] = example

    # for type 3 only
    v['src_pts_doc'] = '\n'.join('{}(float[M]): nonuniform points, valid in [-pi, pi), values outside will be folded'.format(x) for x in pts[:dim])
    v['target_pts_doc'] = '\n'.join('{}(float[N]): nonuniform target points.'.format(x) for x in target_pts[:dim])
    v['pt_inner_type3'] = ' + '.join('{0}[k] {1}[j]'.format(s, x) for s, x in zip(target_pts[:dim], pts[:dim]))

    # for type 3 example only
    v['target_pts'] = ', '.join(str(x) for x in target_pts[:dim])
    v['target_pts_generate'] = '\n'.join('{} = 2 * np.pi * np.random.uniform(size=N)'.format(x) for x in target_pts[:dim])

    if dim > 1:
        v['pt_inner'] = '(' + v['pt_inner'] + ')'
        v['pt_inner_type3'] = '(' + v['pt_inner_type3'] + ')'

    f.__doc__ = _wrap_docstring(doc_nufft[tp].format(**v))


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


_set_nufft_doc(nufft1d1, 1, 1, 'python/finufft/examples/simple1d1.py, python/finufft/examples/simpleopts1d1.py')
_set_nufft_doc(nufft1d2, 1, 2)
_set_nufft_doc(nufft1d3, 1, 3)
_set_nufft_doc(nufft2d1, 2, 1, 'python/finufft/examples/simple2d1.py, python/finufft/examples/many2d1.py')
_set_nufft_doc(nufft2d2, 2, 2)
_set_nufft_doc(nufft2d3, 2, 3)
_set_nufft_doc(nufft3d1, 3, 1)
_set_nufft_doc(nufft3d2, 3, 2)
_set_nufft_doc(nufft3d3, 3, 3)
