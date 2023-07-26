#!/usr/bin/env python
"""
This module contains the high level python wrapper for
the cufinufft CUDA libraries.
"""

import atexit
import inspect
import sys
import warnings

import numpy as np

from ctypes import byref
from ctypes import c_int64
from ctypes import c_void_p

from cufinufft._cufinufft import NufftOpts
from cufinufft._cufinufft import _default_opts
from cufinufft._cufinufft import _make_plan
from cufinufft._cufinufft import _make_planf
from cufinufft._cufinufft import _set_pts
from cufinufft._cufinufft import _set_ptsf
from cufinufft._cufinufft import _exec_plan
from cufinufft._cufinufft import _exec_planf
from cufinufft._cufinufft import _destroy_plan
from cufinufft._cufinufft import _destroy_planf



# If we are shutting down python, we don't need to run __del__
#   This will avoid any shutdown gc ordering problems.
exiting = False
atexit.register(setattr, sys.modules[__name__], 'exiting', True)


class Plan:
    """
    A non-uniform fast Fourier transform (NUFFT) plan

    The ``Plan`` class lets the user exercise more fine-grained control over
    the execution of an NUFFT. First, the plan is created with a certain set
    of parameters (type, mode configuration, tolerance, sign, number of
    simultaneous transforms, and so on). Then the nonuniform points are set
    (source or target depending on the type). Finally, the plan is executed on
    some data, yielding the desired output.

    Args:
        nufft_type      (int): type of NUFFT (1 or 2).
        n_modes         (tuple of ints): the number of modes in each
                        dimension (for example `(50, 100)`).
        n_trans         (int, optional): number of transforms to compute.
        eps             (float, optional): precision requested (>1e-16).
        isign           (int, optional): if +1, uses the positive sign
                        exponential, otherwise the negative sign; defaults to
                        +1 for type 1 and to -1 for type 2.
        dtype           (string, optional): the precision of the transofrm,
                        'complex64' or 'complex128'.
        **kwargs        (optional): additional options corresponding to the
                        entries in the `nufft_opts` structure may be specified
                        as keyword-only arguments.
    """

    def __init__(self, nufft_type, n_modes, n_trans=1, eps=1e-6, isign=None,
                 dtype="complex64", **kwargs):
        if isign is None:
            if nufft_type == 2:
                isign = -1
            else:
                isign = +1

        # Need to set the plan here in case something goes wrong later on,
        # otherwise we error during __del__.
        self._plan = None

        # Setup type bound methods
        self.dtype = np.dtype(dtype)

        if self.dtype == np.float64:
            warnings.warn("Real dtypes are currently deprecated and will be "
                          "removed in version 2.3. Converting to complex128.",
                          DeprecationWarning)
            self.dtype = np.complex128

        if self.dtype == np.float32:
            warnings.warn("Real dtypes are currently deprecated and will be "
                          "removed in version 2.3. Converting to complex64.",
                          DeprecationWarning)
            self.dtype = np.complex64

        if self.dtype == np.complex128:
            self._make_plan = _make_plan
            self._setpts = _set_pts
            self._exec_plan = _exec_plan
            self._destroy_plan = _destroy_plan
            self.real_dtype = np.float64
        elif self.dtype == np.complex64:
            self._make_plan = _make_planf
            self._setpts = _set_ptsf
            self._exec_plan = _exec_planf
            self._destroy_plan = _destroy_planf
            self.real_dtype = np.float32
        else:
            raise TypeError("Expected complex64 or complex128.")

        if isinstance(n_modes, int):
            n_modes = (n_modes,)

        self.dim = len(n_modes)
        self.type = nufft_type
        self.isign = isign
        self.eps = float(eps)
        self.n_modes = n_modes
        self.n_trans = n_trans
        self._maxbatch = 1    # TODO: optimize this one day

        # Get the default option values.
        self._opts = self._default_opts(nufft_type, self.dim)

        # Extract list of valid field names.
        field_names = [name for name, _ in self._opts._fields_]

        # Assign field names from kwargs if they match up, otherwise error.
        for k, v in kwargs.items():
            if k in field_names:
                setattr(self._opts, k, v)
            else:
                raise TypeError(f"Invalid option '{k}'")

        # Initialize the plan.
        self._init_plan()

        # Initialize a list for references to objects
        #   we want to keep around for life of instance.
        self._references = []

    @staticmethod
    def _default_opts(nufft_type, dim):
        """
        Generates a cufinufft opt struct of the dtype coresponding to plan.

        :param nufft_type: integer 1, 2, or 3.
        :param dim: Integer dimension.

        :return: nufft_opts structure.
        """

        nufft_opts = NufftOpts()

        ier = _default_opts(nufft_type, dim, nufft_opts)

        if ier != 0:
            raise RuntimeError('Configuration not yet implemented.')

        return nufft_opts

    def _init_plan(self):
        """
        Internal method to initialize plan struct and call low level make_plan.
        """

        # Initialize struct
        self._plan = c_void_p(None)

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        _n_modes = self.n_modes[::-1] + (1,) * (3 - self.dim)
        _n_modes = (c_int64 * 3)(*_n_modes)

        ier = self._make_plan(self.type,
                              self.dim,
                              _n_modes,
                              self.isign,
                              self.n_trans,
                              self.eps,
                              byref(self._plan),
                              self._opts)

        if ier != 0:
            raise RuntimeError('Error creating plan.')

    def setpts(self, x, y=None, z=None, s=None, t=None, u=None):
        """
        Set the nonuniform points

        For type 1, this sets the coordinates of the ``M`` nonuniform source
        points and for type 2, it sets the coordinates of the ``M`` target
        points.

        The dimension of the plan determines the number of arguments supplied.
        For example, if ``dim == 2``, we provide ``x`` and ``y``.

        Args:
            x       (float[M]): first coordinate of the nonuniform points
                    (source for type 1, target for type 2).
            y       (float[M], optional): second coordinate of the nonuniform
                    points (source for type 1, target for type 2).
            z       (float[M], optional): third coordinate of the nonuniform
                    points (source for type 1, target for type 2).
        """

        _gpu_array_ctor = _get_array_ctor(x)
        _x = _ensure_array_type(x, "x", _gpu_array_ctor, self.real_dtype)
        _y = _ensure_array_type(y, "y", _gpu_array_ctor, self.real_dtype)
        _z = _ensure_array_type(z, "z", _gpu_array_ctor, self.real_dtype)

        _x, _y, _z = _ensure_valid_pts(_x, _y, _z, self.dim)

        if _gpu_array_ctor[1] == 'torch':
            M = len(_x)
        else:
            M = _x.size

        # Because FINUFFT/cufinufft are internally column major,
        #   we will reorder the pts axes. Reordering references
        #   save us from having to actually transpose signal data
        #   from row major (Python default) to column major.
        #   We do this by following translation:
        #     (x, None, None) ~>  (x, None, None)
        #     (x, y, None)    ~>  (y, x, None)
        #     (x, y, z)       ~>  (z, y, x)
        # Via code, we push each dimension onto a stack of axis
        fpts_axes = [_get_ptr(_x), None, None]

        # We will also store references to these arrays.
        #   This keeps python from prematurely cleaning them up.
        self._references.append(_x)
        if self.dim >= 2:
            fpts_axes.insert(0, _get_ptr(_y))
            self._references.append(_y)

        if self.dim >= 3:
            fpts_axes.insert(0, _get_ptr(_z))
            self._references.append(_z)

        # Then take three items off the stack as our reordered axis.
        ier = self._setpts(self._plan, M, *fpts_axes[:3], 0, None, None, None)

        self.nj = M

        if ier != 0:
            raise RuntimeError('Error setting non-uniform points.')

    def execute(self, data, out=None):
        """
        Execute the plan

        Performs the NUFFT specified at plan instantiation with the points set
        by ``setpts``. For type-1 and type-3 transforms, the input is a set of
        source strengths, while for a type-2 transform, it consists of an
        array of size ``n_modes``. If ``n_trans`` is greater than one,
        ``n_trans`` inputs are expected, stacked along the first axis.

        Args:
            data    (complex[M], complex[n_transf, M], complex[n_modes], or complex[n_transf, n_modes]): The input source strengths
                    (type 1) or source modes (type 2).
            out     (complex[n_modes], complex[n_transf, n_modes], complex[M], or complex[n_transf, M], optional): The array where the
                    output is stored. Must be of the right size.

        Returns:
            complex[n_modes], complex[n_transf, n_modes], complex[M], or complex[n_transf, M]: The output array of the transform(s).
        """

        _gpu_array_ctor = _get_array_ctor(data)
        _data = _ensure_array_type(data, "data", _gpu_array_ctor, self.dtype)
        _out = _ensure_array_type(out, "out", _gpu_array_ctor, self.dtype, output=True)

        if self.type == 1:
            req_data_shape = (self.n_trans, self.nj)
            req_out_shape = self.n_modes
        elif self.type == 2:
            req_data_shape = (self.n_trans, *self.n_modes)
            req_out_shape = (self.nj,)

        _data, data_shape = _ensure_array_shape(_data, "data", req_data_shape,
                                                allow_reshape=True)
        if self.type == 1:
            batch_shape = data_shape[:-1]
        else:
            batch_shape = data_shape[:-self.dim]

        req_out_shape = batch_shape + req_out_shape

        if out is None:
            _out = _gpu_array_ctor[0](req_out_shape, dtype=self.dtype)
        else:
            _out = _ensure_array_shape(_out, "out", req_out_shape)

        if self.type == 1:
            ier = self._exec_plan(self._plan, _get_ptr(data), _get_ptr(_out))
        elif self.type == 2:
            ier = self._exec_plan(self._plan, _get_ptr(_out), _get_ptr(data))

        if ier != 0:
            raise RuntimeError('Error executing plan.')

        return _out

    def __del__(self):
        """
        Destroy this instance's associated plan and storage.
        """

        # If the process is exiting or we've already cleaned up plan, return.
        if exiting or self._plan is None:
            return

        ier = self._destroy_plan(self._plan)

        if ier != 0:
            raise RuntimeError('Error destroying plan.')

        # Reset plan to avoid double destroy.
        self._plan = None

        # Reset our reference.
        self._references = []


def _ensure_array_type(x, name, gpu_array_ctor, dtype, output=False):
    if x is None:
        return None

    _, gpu_array_module = gpu_array_ctor
    if gpu_array_module == 'torch':
        if (str(x.dtype) == 'torch.float32' and dtype != np.float32) or (str(x.dtype) == 'torch.float64' and dtype != np.float64):
            raise TypeError(f"Argument `{name}` does not have the correct dtype: "
                            f"{x.dtype} was given, but {dtype} was expected.")
    elif x.dtype != dtype:
        raise TypeError(f"Argument `{name}` does not have the correct dtype: "
                        f"{x.dtype} was given, but {dtype} was expected.")

    if gpu_array_module == 'numba':
        c_contiguous = x.is_c_contiguous()
    elif gpu_array_module == 'torch':
        c_contiguous = x.is_contiguous()
    else:
        c_contiguous = x.flags.c_contiguous

    if not c_contiguous:
        raise TypeError(f"Argument `{name}` does not satisfy the "
                        f"following requirement: C")


    return x


def _ensure_array_shape(x, name, shape, allow_reshape=False):
    orig_shape = x.shape

    if x.shape != shape:
        if not allow_reshape or np.prod(x.shape) != np.prod(shape):
            raise TypeError(f"Argument `{name}` must be of shape {shape}")
        else:
            x = x.reshape(shape)

    if allow_reshape:
        return x, orig_shape
    else:
        return x


def _ensure_valid_pts(x, y, z, dim):
    if x.ndim != 1:
        raise TypeError(f"Argument `x` must be a vector")

    if dim >= 2:
        y = _ensure_array_shape(y, "y", x.shape)

    if dim >= 3:
        z = _ensure_array_shape(z, "z", x.shape)

    if dim < 3 and z and z.size > 0:
        raise TypeError(f"Plan dimension is {dim}, but `z` was specified")

    if dim < 2 and y and y.size > 0:
        raise TypeError(f"Plan dimension is {dim}, but `y` was specified")

    return x, y, z


def _get_ptr(data):
    if not hasattr(data, "__cuda_array_interface__"):
        raise TypeError("Invalid GPU array implementation. Implementation must implement the standard cuda array interface.")
    return data.__cuda_array_interface__['data'][0]


def _get_module(obj):
    return inspect.getmodule(type(obj)).__name__


def _get_array_ctor(obj):
    module_name = _get_module(obj)
    if module_name.startswith('numba.cuda'):
        import numba.cuda
        return (numba.cuda.device_array, 'numba')
    elif module_name.startswith('torch'):
        import torch
        def ctor(*args, **kwargs):
            if 'shape' in kwargs:
                kwargs['size'] = kwargs.pop('shape')
            if 'dtype' in kwargs:
                dtype = kwargs.pop('dtype')
                if dtype == np.complex64:
                    dtype = torch.complex64
                if dtype == np.complex128:
                    dtype = torch.complex128
                kwargs['dtype'] = dtype
            if 'device' not in kwargs:
                kwargs['device'] = obj.device

            return torch.empty(*args, **kwargs)

        return (ctor, 'torch')
    else:
        return (type(obj), 'generic')
