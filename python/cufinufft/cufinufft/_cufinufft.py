#!/usr/bin/env python
"""
This file contains low level python bindings for the cufinufft CUDA libraries.
Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""

import ctypes
import os
import warnings
import importlib.util
import pathlib
import numpy as np
from ctypes.util import find_library

from packaging.version import Version

from ctypes import c_double
from ctypes import c_int
from ctypes import c_int64
from ctypes import c_float
from ctypes import c_void_p

c_int64_p = ctypes.POINTER(c_int64)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)


# numpy.distutils has a bug that changes the logging level from under us. As a
# workaround, we save it and reset it later. This only happens on older
# versions of NumPy, so let's check the version before doing this.
reset_log_level = Version(np.__version__) < Version("1.25")

if reset_log_level:
    import logging
    log_level = logging.root.level

lib = None
# Try to load the library as installed in the Python package
path = pathlib.Path(__file__).parent.resolve()
library_names = ["libcufinufft", "cufinufft"]

if reset_log_level:
    logging.root.setLevel(log_level)

# First attempt: try from package directory
for lib_name in library_names:
    try:
        lib = np.ctypeslib.load_library(lib_name, path)
        break
    except (OSError, AttributeError):
        pass

# Second attempt: try from system path
if lib is None:
    libname = find_library('cufinufft')
    if libname is None:
        raise ImportError("Could not find cufinufft library in system path")
    try:
        lib = ctypes.cdll.LoadLibrary(libname)
    except (OSError, AttributeError) as e:
        raise ImportError(f"Found cufinufft library at {libname}, but failed to load it: {e}")

# Safety check - if somehow we still don't have a library
if lib is None:
    raise ImportError("Failed to load cufinufft library")

def _get_NufftOpts():
    fields = [
        ('upsampfac', c_double),
        ('gpu_method', c_int),
        ('gpu_sort', c_int),
        ('gpu_binsizex', c_int),
        ('gpu_binsizey', c_int),
        ('gpu_binsizez', c_int),
        ('gpu_obinsizex', c_int),
        ('gpu_obinsizey', c_int),
        ('gpu_obinsizez', c_int),
        ('gpu_maxsubprobsize', c_int),
        ('gpu_kerevalmeth', c_int),
        ('gpu_spreadinterponly', c_int),
        ('gpu_maxbatchsize', c_int),
        ('gpu_device_id', c_int),
        ('gpu_stream', c_void_p),
        ('modeord', c_int)
    ]
    return fields


class NufftOpts(ctypes.Structure):
    pass


NufftOpts._fields_ = _get_NufftOpts()


CufinufftPlan = c_void_p
CufinufftPlanf = c_void_p

CufinufftPlan_p = ctypes.POINTER(CufinufftPlan)
CufinufftPlanf_p = ctypes.POINTER(CufinufftPlanf)

NufftOpts_p = ctypes.POINTER(NufftOpts)

_default_opts = lib.cufinufft_default_opts
_default_opts.argtypes = [NufftOpts_p]
_default_opts.restype = None

_make_plan = lib.cufinufft_makeplan
_make_plan.argtypes = [
    c_int, c_int, c_int64_p, c_int,
    c_int, c_double, CufinufftPlan_p, NufftOpts_p]
_make_plan.restypes = c_int

_make_planf = lib.cufinufftf_makeplan
_make_planf.argtypes = [
    c_int, c_int, c_int64_p, c_int,
    c_int, c_float, CufinufftPlanf_p, NufftOpts_p]
_make_planf.restypes = c_int

_set_pts = lib.cufinufft_setpts
_set_pts.argtypes = [
    c_void_p, c_int64, c_void_p, c_void_p, c_void_p, ctypes.c_int64, c_void_p,
    c_void_p, c_void_p]
_set_pts.restype = c_int

_set_ptsf = lib.cufinufftf_setpts
_set_ptsf.argtypes = [
    c_void_p, c_int64, c_void_p, c_void_p, c_void_p, ctypes.c_int64, c_void_p,
    c_void_p, c_void_p]
_set_ptsf.restype = c_int

_exec_plan = lib.cufinufft_execute
_exec_plan.argtypes = [c_void_p, c_void_p, c_void_p]
_exec_plan.restype = c_int

_exec_planf = lib.cufinufftf_execute
_exec_planf.argtypes = [c_void_p, c_void_p, c_void_p]
_exec_planf.restype = c_int

_destroy_plan = lib.cufinufft_destroy
_destroy_plan.argtypes = [c_void_p]
_destroy_plan.restype = c_int

_destroy_planf = lib.cufinufftf_destroy
_destroy_planf.argtypes = [c_void_p]
_destroy_planf.restype = c_int
