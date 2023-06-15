#!/usr/bin/env python
"""
This module contains the high level python wrapper for
the cufinufft CUDA libraries.
"""

import atexit
import sys

import numpy as np

from ctypes import byref
from ctypes import c_int
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

from pycuda.gpuarray import GPUArray


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
        modes           (tuple of ints): the number of modes in each
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

    def __init__(self, nufft_type, modes, n_trans=1, eps=1e-6, isign=None,
                 dtype="complex64", **kwargs):
        if isign is None:
            if nufft_type == 2:
                isign = -1
            else:
                isign = +1

        # Need to set the plan here in case something goes wrong later on,
        # otherwise we error during __del__.
        self.plan = None

        # Setup type bound methods
        self.dtype = np.dtype(dtype)

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

        self.dim = len(modes)
        self._finufft_type = nufft_type
        self.isign = isign
        self.eps = float(eps)
        self.n_trans = n_trans
        self._maxbatch = 1    # TODO: optimize this one day
        self.modes = modes

        # Get the default option values.
        self.opts = self._default_opts(nufft_type, self.dim)

        # Extract list of valid field names.
        field_names = [name for name, _ in self.opts._fields_]

        # Assign field names from kwargs if they match up, otherwise error.
        for k, v in kwargs.items():
            if k in field_names:
                setattr(self.opts, k, v)
            else:
                raise TypeError(f"Invalid option '{k}'")

        # Initialize the plan.
        self._plan()

        # Initialize a list for references to objects
        #   we want to keep around for life of instance.
        self.references = []

    @staticmethod
    def _default_opts(nufft_type, dim):
        """
        Generates a cufinufft opt struct of the dtype coresponding to plan.

        :param finufft_type: integer 1, 2, or 3.
        :param dim: Integer dimension.

        :return: nufft_opts structure.
        """

        nufft_opts = NufftOpts()

        ier = _default_opts(nufft_type, dim, nufft_opts)

        if ier != 0:
            raise RuntimeError('Configuration not yet implemented.')

        return nufft_opts

    def _plan(self):
        """
        Internal method to initialize plan struct and call low level make_plan.
        """

        # Initialize struct
        self.plan = c_void_p(None)

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        _modes = self.modes[::-1] + (1,) * (3 - self.dim)
        _modes = (c_int * 3)(*_modes)

        ier = self._make_plan(self._finufft_type,
                              self.dim,
                              _modes,
                              self.isign,
                              self.n_trans,
                              self.eps,
                              1,
                              byref(self.plan),
                              self.opts)

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

        if x.dtype != self.real_dtype:
            raise TypeError("cufinufft plan.real_dtype and "
                            "x dtypes do not match.")

        if y is not None and y.dtype != self.real_dtype:
            raise TypeError("cufinufft plan.real_dtype and "
                            "y dtypes do not match.")

        if z is not None and z.dtype != self.real_dtype:
            raise TypeError("cufinufft plan.real_dtype and "
                            "z dtypes do not match.")

        M = x.size

        if y is not None and y.size != M:
            raise TypeError("Number of elements in x and y must be equal")

        if z is not None and z.size != M:
            raise TypeError("Number of elements in x and z must be equal")

        # Because FINUFFT/cufinufft are internally column major,
        #   we will reorder the pts axes. Reordering references
        #   save us from having to actually transpose signal data
        #   from row major (Python default) to column major.
        #   We do this by following translation:
        #     (x, None, None) ~>  (x, None, None)
        #     (x, y, None)    ~>  (y, x, None)
        #     (x, y, z)       ~>  (z, y, x)
        # Via code, we push each dimension onto a stack of axis
        fpts_axes = [x.ptr, None, None]

        # We will also store references to these arrays.
        #   This keeps python from prematurely cleaning them up.
        self.references.append(x)
        if y is not None:
            fpts_axes.insert(0, y.ptr)
            self.references.append(y)

        if z is not None:
            fpts_axes.insert(0, z.ptr)
            self.references.append(z)

        # Then take three items off the stack as our reordered axis.
        ier = self._setpts(M, *fpts_axes[:3], 0, None, None, None, self.plan)

        self.n_pts = M

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

        if data.dtype != self.dtype:
            raise TypeError("execute expects {} dtype arguments for this "
                            "plan. Check plan and arguments.".format(
                                self.dtype))

        if out is not None and out.dtype != self.dtype:
            raise TypeError("execute expects {} dtype arguments for this "
                            "plan. Check plan and arguments.".format(
                                self.dtype))

        if out is None:
            if self._finufft_type == 1:
                _out = GPUArray((self.n_trans, *self.modes), dtype=self.dtype)
            elif self._finufft_type == 2:
                _out = GPUArray((self.n_trans, self.n_pts), dtype=self.dtype)
        else:
            _out = out

        if self._finufft_type == 1:
            ier = self._exec_plan(data.ptr, _out.ptr, self.plan)
        elif self._finufft_type == 2:
            ier = self._exec_plan(_out.ptr, data.ptr, self.plan)

        if ier != 0:
            raise RuntimeError('Error executing plan.')

        return _out

    def __del__(self):
        """
        Destroy this instance's associated plan and storage.
        """

        # If the process is exiting or we've already cleaned up plan, return.
        if exiting or self.plan is None:
            return

        ier = self._destroy_plan(self.plan)

        if ier != 0:
            raise RuntimeError('Error destroying plan.')

        # Reset plan to avoid double destroy.
        self.plan = None

        # Reset our reference.
        self.references = []
