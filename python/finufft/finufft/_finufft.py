#!/usr/bin/env python
"""
Adapted from cufinufft python interface using ctypes.
This file contains low level python bindings for the finufft libraries.
Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""

import ctypes
import os
import warnings
import platform

from ctypes import c_double
from ctypes import c_int
from ctypes import c_float
from ctypes import c_void_p
from ctypes import c_longlong
from numpy.ctypeslib import ndpointer

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)
c_longlong_p = ctypes.POINTER(c_longlong)

# Attempt to load library, first from package install, then from path as fallback
try:
    pkgroot = os.path.dirname(__file__)
    if platform.system() == 'Windows':
        os.environ["PATH"] += os.pathsep + pkgroot
        basename = "libfinufft.dll"
    else:
        basename = "libfinufft.so"

    full_lib_path = os.path.join(pkgroot, basename)
    try:
        lib = ctypes.cdll.LoadLibrary(full_lib_path)
    except:
        lib = ctypes.cdll.LoadLibrary(basename)

except Exception:
    raise ImportError('Failed to find a suitable finufft library')


class FinufftOpts(ctypes.Structure):
    pass


FinufftOpts._fields_ = [('modeord', c_int),
                      ('chkbnds', c_int),
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
                      ('spread_max_sp_size', c_int)]


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
