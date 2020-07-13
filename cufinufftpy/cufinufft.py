#!/usr/bin/env python
"""
This module contains the high level python wrapper for
the cufinufft CUDA libraries.
"""

import numpy as np

from ctypes import c_int

from cufinufftpy._cufinufft import NufftOpts
from cufinufftpy._cufinufft import NufftOptsf
from cufinufftpy._cufinufft import CufinufftPlan
from cufinufftpy._cufinufft import CufinufftPlanf
from cufinufftpy._cufinufft import _default_opts
from cufinufftpy._cufinufft import _default_optsf
from cufinufftpy._cufinufft import _make_plan
from cufinufftpy._cufinufft import _make_planf
from cufinufftpy._cufinufft import _set_nu_pts
from cufinufftpy._cufinufft import _set_nu_ptsf
from cufinufftpy._cufinufft import _exec_plan
from cufinufftpy._cufinufft import _exec_planf
from cufinufftpy._cufinufft import _destroy_plan
from cufinufftpy._cufinufft import _destroy_planf


class cufinufft:
    """
    Upon instantiation of a cufinufft instance, dtype of `modes` is detected.
    This dtype selects which of the low level libraries to bind for this plan.
    The wrapper performs a few very basic conversions,
    and calls the low level library with runtime python error checking.
    """
    def __init__(self, nufft_type, modes, isign, tol,
                 ntransforms=1, opts=None, dtype=np.float32):
        """
        Initialize a dtype bound cufinufft python wrapper.
        This will bind variables/methods
        and make a plan with the cufinufft libraries.
        Exposes python methods to execute and destroy.

        :param finufft_type: integer 1, 2, or 3.
        :param modes: Array describing the shape of the transform \
        in 1, 2, or 3 dimensions.
        :param isign: 1 or -1, controls sign of imaginary component output.
        :param tol: Floating point tolerance.
        :param ntransforms: Number of transforms, defaults to 1.
        :param opts: Optionally, supply opts struct (untested).
        :param dtype: Datatype for this plan (np.float32 or np.float64). \
        Defaults np.float32.

        :return: cufinufft instance of the correct dtype, \
        ready for point setting, and execution.
        """

        # Setup type bound methods
        self.dtype = np.dtype(dtype)

        if self.dtype == np.float64:
            self.Nufft_Opts = NufftOpts
            self.CufinufftPlan = CufinufftPlan
            self._default_opts = _default_opts
            self._make_plan = _make_plan
            self._set_nu_pts = _set_nu_pts
            self._exec_plan = _exec_plan
            self._destroy_plan = _destroy_plan
            self.complex_dtype = np.complex128
        elif self.dtype == np.float32:
            self.Nufft_Opts = NufftOptsf
            self.CufinufftPlan = CufinufftPlanf
            self._default_opts = _default_optsf
            self._make_plan = _make_planf
            self._set_nu_pts = _set_nu_ptsf
            self._exec_plan = _exec_planf
            self._destroy_plan = _destroy_planf
            self.complex_dtype = np.complex64
        else:
            raise TypeError("Expected np.float32 or np.float64.")

        self.dim = len(modes)
        self._finufft_type = nufft_type
        self.isign = isign
        self.tol = float(tol)
        self.ntransforms = ntransforms
        self._maxbatch = 1    # TODO: optimize this one day

        # Setup Options
        if opts is None:
            opts = self.default_opts(nufft_type, self.dim)
        self.opts = opts

        modes = modes + (1,) * (3 - self.dim)
        modes = (c_int * 3)(*modes)
        self.modes = modes

        # Initialize the plan for this instance
        self._plan()

    def default_opts(self, nufft_type, dim):
        """
        Generates a cufinufft opt struct of the dtype coresponding to plan.

        :param finufft_type: integer 1, 2, or 3.
        :param dim: Integer dimension.

        :return: nufft_opts structure.
        """

        nufft_opts = self.Nufft_Opts()

        ier = self._default_opts(nufft_type, dim, nufft_opts)

        if ier != 0:
            raise RuntimeError('Configuration not yet implemented.')

        return nufft_opts

    def _plan(self):
        """
        Internal method to initialize plan struct and call low level make_plan.
        """

        # Initialize struct
        plan = self.CufinufftPlan()
        plan.opts = self.opts

        ier = self._make_plan(self._finufft_type,
                              self.dim,
                              self.modes,
                              self.isign,
                              self.ntransforms,
                              self.tol,
                              1,
                              plan)

        if ier != 0:
            raise RuntimeError('Error creating plan.')

        self.plan = plan

    def set_nu_pts(self, M, kx, ky=None, kz=None):
        """
        Sets non uniform points of the correct dtype.

        Note kx, ky, kz are required for 1, 2, and 3
        dimensional cases respectively.

        :param M: Number of points
        :param kx: Array of x points.
        :param ky: Array of y points.
        :param kz: Array of z points.
        """

        if not (kx.dtype == ky.dtype == kz.dtype == self.dtype):
            raise TypeError("cifinufft plan.dtype and "
                            "kx, ky, kz dtypes do not match.")

        kx = kx.ptr

        if ky is not None:
            ky = ky.ptr

        if kz is not None:
            kz = kz.ptr

        ier = self._set_nu_pts(M, kx, ky, kz, 0, None, None, None, self.plan)

        if ier != 0:
            raise RuntimeError('Error setting non-uniform points.')

    def execute(self, c, fk):
        """
        Executes plan. Note the IO orientation of `c` and `fk` are
        determined by plan type.

        In type 1, `c` is input, `fk` is output, while
        in type 2, 'fk' in input, `c` is output.

        :param c: Real space array in 1, 2, or 3 dimensions.
        :param fk: Fourier space array in 1, 2, or 3 dimensions.
        """

        if not c.dtype == fk.dtype == self.complex_dtype:
            raise TypeError("cufinufft execute expects {} dtype arguments "
                            "for this plan. Check plan and arguments.".format(
                                self.complex_dtype))

        ier = self._exec_plan(c.ptr, fk.ptr, self.plan)

        if ier != 0:
            raise RuntimeError('Error executing plan.')

    def destroy(self):
        """
        Destroy this instance's associated plan and storage.
        """

        ier = self._destroy_plan(self.plan)

        if ier != 0:
            raise RuntimeError('Error destroying plan.')
