#!/usr/bin/env python
"""
Adapted from cufinufft python interface using ctypes.
This file contains low level python bindings for the finufft libraries.
Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""
import ctypes
import pathlib
from ctypes.util import find_library
from ctypes import c_double
from ctypes import c_float
from ctypes import c_int
from ctypes import c_longlong
from ctypes import c_void_p

import numpy as np
import os
import platform
from numpy.ctypeslib import ndpointer
import logging


from packaging.version import Version

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)
c_longlong_p = ctypes.POINTER(c_longlong)

# numpy.distutils has a bug that changes the logging level from under us. As a
# workaround, we save it and reset it later. This only happens on older
# versions of NumPy, so let's check the version before doing this.
reset_log_level = Version(np.__version__) < Version("1.25")

if reset_log_level:
    import logging
    log_level = logging.root.level

lib = None
log_level = logging.root.level
reset_log_level = logging.getLogger().getEffectiveLevel() != logging.NOTSET

# Try to load local libfinufft.so next to this file
path = pathlib.Path(__file__).parent.resolve()
for lib_name in ['libfinufft', 'finufft']:
    candidate = path / f"{lib_name}.so"
    if candidate.exists():
        try:
            lib = ctypes.CDLL(str(candidate))
            break
        except OSError:
            lib = None

# Try system path as fallback
if lib is None:
    libname = find_library('finufft')
    if libname:
        try:
            lib = ctypes.CDLL(libname)
        except OSError:
            lib = None

if reset_log_level:
    logging.root.setLevel(log_level)

if lib is None:
    raise ImportError(
        'Failed to find a suitable finufft library. '
        'Please check your installation.'
    )

class FinufftOpts(ctypes.Structure):
    pass


FinufftOpts._fields_ = [('modeord', c_int),
                      ('spreadinterponly', c_int),
                      ('debug', c_int),
                      ('spread_debug', c_int),
                      ('showwarn', c_int),
                      ('nthreads', c_int),
                      ('fftw', c_int),
                      ('spread_sort', c_int),
                      ('spread_kerevalmeth', c_int),
                      ('spread_kerpad', c_int),
                      ('upsampfac', c_double),
                      ('spread_thread', c_int),
                      ('maxbatchsize', c_int),
                      ('spread_nthr_atomic', c_int),
                      ('spread_max_sp_size', c_int),
                      ('fftw_lock_fun', c_void_p),
                      ('fftw_unlock_fun', c_void_p),
                      ('fftw_lock_data', c_void_p)]


FinufftPlan = c_void_p
FinufftPlanf = c_void_p

FinufftPlan_p = ctypes.POINTER(FinufftPlan)
FinufftPlanf_p = ctypes.POINTER(FinufftPlanf)

FinufftOpts_p = ctypes.POINTER(FinufftOpts)

_default_opts = lib.finufft_default_opts
_default_opts.argtypes = [FinufftOpts_p]
_default_opts.restype = None

_makeplan = lib.finufft_makeplan
_makeplan.argtypes = [
    c_int, c_int, c_longlong_p, c_int,
    c_int, c_double, FinufftPlan_p, FinufftOpts_p]
_makeplan.restypes = c_int

_makeplanf = lib.finufftf_makeplan
_makeplanf.argtypes = [
    c_int, c_int, c_longlong_p, c_int,
    c_int, c_float, FinufftPlanf_p, FinufftOpts_p]
_makeplanf.restypes = c_int

_setpts = lib.finufft_setpts
_setpts.argtypes = [
    FinufftPlan, c_longlong, ndpointer(c_double), ndpointer(c_double), ndpointer(c_double),
    ctypes.c_longlong, ndpointer(c_double), ndpointer(c_double), ndpointer(c_double)]
_setpts.restype = c_int

_setptsf = lib.finufftf_setpts
_setptsf.argtypes = [
    FinufftPlanf, c_longlong, ndpointer(c_float), ndpointer(c_float), ndpointer(c_float),
    ctypes.c_longlong, ndpointer(c_float), ndpointer(c_float), ndpointer(c_float)]
_setptsf.restype = c_int

_execute = lib.finufft_execute
_execute.argtypes = [c_void_p, c_void_p, c_void_p]
_execute.restype = c_int

_executef = lib.finufftf_execute
_executef.argtypes = [c_void_p, c_void_p, c_void_p]
_executef.restype = c_int

_destroy = lib.finufft_destroy
_destroy.argtypes = [c_void_p]
_destroy.restype = c_int

_destroyf = lib.finufftf_destroy
_destroyf.argtypes = [c_void_p]
_destroyf.restype = c_int
