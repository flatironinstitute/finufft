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

# While imp is deprecated, it is currently the inspection solution
#   that works for all versions of Python 2 and 3.
# One day if that changes, can be replaced
#   with importlib.find_spec.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

import numpy as np

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

# TODO: See if there is a way to improve this so it is less hacky.
lib = None
# Try to load a local library directly.
try:
    lib = ctypes.cdll.LoadLibrary('libfinufft.so')
except OSError:
    pass

# Should that not work, try to find the full path of a packaged lib.
#   The packaged lib should have a py/platform decorated name,
#   and be rpath'ed the true FINUFFT library through the Extension and wheel
#   systems.
try:
    if lib is None:
        # Find the library.
        fh = imp.find_module('finufft/finufftc')[0]
        # Get the full path for the ctypes loader.
        if platform.system() == 'Windows':
            os.environ["PATH"] += os.pathsep + os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(fh.name))),'finufft')
            full_lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(fh.name))),'finufft','libfinufft.dll')
        else:
            full_lib_path = os.path.realpath(fh.name)
        fh.close()    # Be nice and close the open file handle.

        # Load the library,
        #    which rpaths the libraries we care about.
        lib = ctypes.cdll.LoadLibrary(full_lib_path)
except Exception:
    raise RuntimeError('Failed to find a suitable finufft library')


class NufftOpts(ctypes.Structure):
    pass


NufftOpts._fields_ = [('modeord', c_int),
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

NufftOpts_p = ctypes.POINTER(NufftOpts)

_default_opts = lib.finufft_default_opts
_default_opts.argtypes = [NufftOpts_p]
_default_opts.restype = None

_makeplan = lib.finufft_makeplan
_makeplan.argtypes = [
    c_int, c_int, c_longlong_p, c_int,
    c_int, c_double, FinufftPlan_p, NufftOpts_p]
_makeplan.restypes = c_int

_makeplanf = lib.finufftf_makeplan
_makeplanf.argtypes = [
    c_int, c_int, c_longlong_p, c_int,
    c_int, c_float, FinufftPlanf_p, NufftOpts_p]
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
