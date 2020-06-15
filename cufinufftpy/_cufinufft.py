#!/usr/bin/env python
"""
This file contains low level python bindings for the cufinufft CUDA libraries.
Seperate bindings are provided for single and double precision libraries,
differentiated by 'f' suffix.
"""

import ctypes

import numpy as np
import pycuda.driver as cuda

from ctypes import c_double
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_float
from ctypes import c_void_p

c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)
c_double_p = ctypes.POINTER(c_double)

# TODO: Thinking about how to make this safer and more portable.
lib = ctypes.cdll.LoadLibrary('libcufinufftc.so')
libf = ctypes.cdll.LoadLibrary('libcufinufftcf.so')

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

def _get_NufftOptsFields(dtype):
    REAL_t, REAL_ptr = _get_ctypes(dtype)
    fields = [
        ('upsampfac', REAL_t),
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

class NufftOpts(ctypes.Structure): pass
NufftOpts._fields_ = _get_NufftOptsFields(np.float64)

class NufftOptsf(ctypes.Structure): pass
NufftOptsf._fields_ = _get_NufftOptsFields(np.float32)

def _get_NufftOpts(dtype):
    if dtype == np.float64:
        s = NufftOpts
    elif dtype == np.float32:
        s = NufftOptsf
    else:
        raise TypeError("Expected np.float32 or np.float64.")

    return s

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


class SpreadOpts(ctypes.Structure): pass
SpreadOpts._fields_ = _get_SpeadOptsFields(np.float64)

class SpreadOptsf(ctypes.Structure): pass
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
        ('opts', _get_NufftOpts(dtype)),
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


class CufinufftPlan(ctypes.Structure): pass
CufinufftPlan._fields_ = _get_CufinufftPlan(np.float64)

class CufinufftPlanf(ctypes.Structure): pass
CufinufftPlanf._fields_ = _get_CufinufftPlan(np.float32)

CufinufftPlan_p = ctypes.POINTER(CufinufftPlan)
CufinufftPlanf_p = ctypes.POINTER(CufinufftPlanf)

NufftOpts_p = ctypes.POINTER(NufftOpts)
NufftOptsf_p = ctypes.POINTER(NufftOptsf)

_default_opts = lib.cufinufftc_default_opts
_default_opts.argtypes = [c_int, c_int, NufftOpts_p]
_default_opts.restype = c_int

_default_optsf = libf.cufinufftc_default_opts
_default_optsf.argtypes = [c_int, c_int, NufftOptsf_p]
_default_optsf.restype = c_int

_make_plan = lib.cufinufftc_makeplan
_make_plan.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_double, c_int, CufinufftPlan_p]
_make_plan.restypes = c_int

_make_planf = libf.cufinufftc_makeplan
_make_planf.argtypes = [
    c_int, c_int, c_int_p, c_int,
    c_int, c_float, c_int, CufinufftPlanf_p]
_make_planf.restypes = c_int

_set_nu_pts = lib.cufinufftc_setNUpts
_set_nu_pts.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_double_p,
    c_double_p, c_double_p, CufinufftPlan_p]
_set_nu_pts.restype = c_int

_set_nu_ptsf = libf.cufinufftc_setNUpts
_set_nu_ptsf.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_float_p,
    c_float_p, c_float_p, CufinufftPlanf_p]
_set_nu_ptsf.restype = c_int

_exec_plan = lib.cufinufftc_exec
_exec_plan.argtypes = [c_void_p, c_void_p, CufinufftPlan_p]
_exec_plan.restype = c_int

_exec_planf = libf.cufinufftc_exec
_exec_planf.argtypes = [c_void_p, c_void_p, CufinufftPlanf_p]
_exec_planf.restype = c_int

_destroy_plan = lib.cufinufftc_destroy
_destroy_plan.argtypes = [CufinufftPlan_p]
_destroy_plan.restype = c_int

_destroy_planf = libf.cufinufftc_destroy
_destroy_planf.argtypes = [CufinufftPlanf_p]
_destroy_planf.restype = c_int
