#!/usr/bin/env python
"""
This file contains low level python bindings for the cufinufft CUDA libraries.
Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""

import ctypes
import os
import warnings

# While imp is deprecated, it is currently the inspection solution
#   that works for all versions of Python 2 and 3.
# One day if that changes, can be replaced
#   with importlib.find_spec.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import imp

import numpy as np

from ctypes import c_double
from ctypes import c_int
from ctypes import c_float
from ctypes import c_void_p

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

# TODO: See if there is a way to improve this so it is less hacky.
lib = None
# Try to load a local library directly.
try:
    lib = ctypes.cdll.LoadLibrary('libcufinufftc.so')
except OSError as e:
    pass

# Should that not work, try to find the full path of a packaged lib.
#   The packaged lib should have a py/platform decorated name,
#   and be rpath'ed the true CUDA C cufinufftc library through the
#   Extension and wheel systems.
try:
    if lib is None:
        # Find the library.
        fh = imp.find_module('cufinufftc')[0]
        # Get the full path for the ctypes loader.
        full_lib_path =  os.path.realpath(fh.name)
        fh.close()    # Be nice and close the open file handle.

        # Load the library,
        #    which rpaths the libraries we care about.
        lib = ctypes.cdll.LoadLibrary(full_lib_path)

except Exception as e:
    raise RuntimeError('Failed to find a suitable cufinufftc library')



def _get_ctypes(dtype):
    """
    Checks dtype is float32 or float64.
    Returns floating point and floating point pointer.
    """

    if dtype == np.float64:
        REAL_t = c_double
    elif dtype == np.float32:
        REAL_t = c_float
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    REAL_ptr = ctypes.POINTER(REAL_t)

    return REAL_t, REAL_ptr


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
        ('gpu_nstreams', c_int),
        ('gpu_kerevalmeth', c_int)]
    return fields


class NufftOpts(ctypes.Structure):
    pass


NufftOpts._fields_ = _get_NufftOpts()


def _get_SpeadOptsFields(dtype):
    REAL_t, REAL_ptr = _get_ctypes(dtype)
    fields = [
        ('nspread', c_int),
        ('spread_direction', c_int),
        ('pirange', c_int),
        ('upsampfac', REAL_t),
        ('ES_beta', REAL_t),
        ('ES_halfwidth', REAL_t),
        ('ES_c', REAL_t)]
    return fields


class SpreadOpts(ctypes.Structure):
    pass


SpreadOpts._fields_ = _get_SpeadOptsFields(np.float64)


class SpreadOptsf(ctypes.Structure):
    pass


SpreadOptsf._fields_ = _get_SpeadOptsFields(np.float32)


def _get_SpreadOpts(dtype):
    if dtype == np.float64:
        s = SpreadOpts
    elif dtype == np.float32:
        s = SpreadOptsf
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    return s


def _get_CufinufftPlan(dtype):
    REAL_t, REAL_ptr = _get_ctypes(dtype)

    fields = [
        ('opts', NufftOpts),
        ('spopts', _get_SpreadOpts(dtype)),
        ('type', c_int),
        ('dim', c_int),
        ('M', c_int),
        ('nf1', c_int),
        ('nf2', c_int),
        ('nf3', c_int),
        ('ms', c_int),
        ('mt', c_int),
        ('mu', c_int),
        ('ntransf', c_int),
        ('maxbatchsize', c_int),
        ('iflag', c_int),
        ('totalnumsubprob', c_int),
        ('byte_now', c_int),
        ('fwkerhalf1', REAL_ptr),
        ('fwkerhalf2', REAL_ptr),
        ('fwkerhalf3', REAL_ptr),
        ('kx', REAL_ptr),
        ('ky', REAL_ptr),
        ('kz', REAL_ptr),
        ('c', c_void_p),
        ('fw', c_void_p),
        ('fk', c_void_p),
        ('idxnupts', c_int_p),
        ('sortidx', c_int_p),
        ('numsubprob', c_int_p),
        ('binsize', c_int_p),
        ('binstartpts', c_int_p),
        ('subprob_to_bin', c_int_p),
        ('subprobstartpts', c_int_p),
        ('finegridsize', c_int_p),
        ('fgstartpts', c_int_p),
        ('numnupts', c_int_p),
        ('subprob_to_nupts', c_int_p),
        ('fftplan', c_int),
        ('streams', c_void_p)]

    return fields


class CufinufftPlan(ctypes.Structure):
    pass


CufinufftPlan._fields_ = _get_CufinufftPlan(np.float64)


class CufinufftPlanf(ctypes.Structure):
    pass


CufinufftPlanf._fields_ = _get_CufinufftPlan(np.float32)

CufinufftPlan_p = ctypes.POINTER(CufinufftPlan)
CufinufftPlanf_p = ctypes.POINTER(CufinufftPlanf)

NufftOpts_p = ctypes.POINTER(NufftOpts)

_default_opts = lib.cufinufftc_default_opts
_default_opts.argtypes = [c_int, c_int, NufftOpts_p]
_default_opts.restype = c_int

_make_plan = lib.cufinufftc_makeplan
_make_plan.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_double, c_int, CufinufftPlan_p, NufftOpts_p]
_make_plan.restypes = c_int

_make_planf = lib.cufinufftcf_makeplan
_make_planf.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_float, c_int, CufinufftPlanf_p, NufftOpts_p]
_make_planf.restypes = c_int

_set_pts = lib.cufinufftc_setpts
_set_pts.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_double_p,
    c_double_p, c_double_p, CufinufftPlan_p]
_set_pts.restype = c_int

_set_ptsf = lib.cufinufftcf_setpts
_set_ptsf.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_float_p,
    c_float_p, c_float_p, CufinufftPlanf_p]
_set_ptsf.restype = c_int

_exec_plan = lib.cufinufftc_exec
_exec_plan.argtypes = [c_void_p, c_void_p, CufinufftPlan_p]
_exec_plan.restype = c_int

_exec_planf = lib.cufinufftcf_exec
_exec_planf.argtypes = [c_void_p, c_void_p, CufinufftPlanf_p]
_exec_planf.restype = c_int

_destroy_plan = lib.cufinufftc_destroy
_destroy_plan.argtypes = [CufinufftPlan_p]
_destroy_plan.restype = c_int

_destroy_planf = lib.cufinufftcf_destroy
_destroy_planf.argtypes = [CufinufftPlanf_p]
_destroy_planf.restype = c_int
