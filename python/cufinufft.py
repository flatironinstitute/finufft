import ctypes

import numpy as np
import pycuda.driver as cuda

c_int = ctypes.c_int
c_uint = ctypes.c_uint
c_float = ctypes.c_float

c_void_p = ctypes.c_void_p
c_int_p = ctypes.POINTER(c_int)
c_float_p = ctypes.POINTER(c_float)

class NufftOpts(ctypes.Structure): pass
NufftOpts._fields_ = [
    ('debug', c_int),
    ('spread_debug', c_int),
    ('spread_sort', c_int),
    ('spread_kerevalmeth', c_int),
    ('spread_kerpad', c_int),
    ('chkbnds', c_int),
    ('fftw', c_int),
    ('modeord', c_int),
    ('upsampfac', c_float),
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

class SpreadOpts(ctypes.Structure): pass
SpreadOpts._fields_ = [
    ('nspread', c_int),
    ('spread_direction', c_int),
    ('pirange', c_int),
    ('chkbnds', c_int),
    ('sort', c_int),
    ('kerevalmeth', c_int),
    ('kerpad', c_int),
    ('sort_threads', c_int),
    ('max_subproblem_size', c_int),
    ('flags', c_int),
    ('debug', c_int),
    ('upsampfac', c_float),
    ('ES_beta', c_float),
    ('ES_halfwidth', c_float),
    ('ES_c', c_float)]

class CufinufftPlan(ctypes.Structure): pass
CufinufftPlan._fields_ = [
    ('type', c_uint),
    ('opts', NufftOpts),
    ('spopts', SpreadOpts),
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
    ('fwkerhalf1', c_float_p),
    ('fwkerhalf2', c_float_p),
    ('fwkerhalf3', c_float_p),
    ('kx', c_float_p),
    ('ky', c_float_p),
    ('kz', c_float_p),
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

CufinufftPlan_p = ctypes.POINTER(CufinufftPlan)
NufftOpts_p = ctypes.POINTER(NufftOpts)

lib = ctypes.cdll.LoadLibrary('../lib/libcufinufftcf.so')

_default_opts = lib.cufinufftc_default_opts
_default_opts.argtypes = [c_uint, c_int, NufftOpts_p]
_default_opts.restype = c_int


def default_opts(finufft_type, dim):
    nufft_opts = NufftOpts()

    ier = _default_opts(finufft_type - 1, dim, nufft_opts)

    if ier != 0:
        raise RuntimeError('Configuration not yet implemented.')

    return nufft_opts

_make_plan = lib.cufinufftc_makeplan
_make_plan.argtypes = [
    c_uint, c_int, c_int_p, c_int,
    c_int, c_float, c_int, CufinufftPlan_p]
_make_plan.restypes = c_int

def plan(finufft_type, modes, isign, tol, opts=None):
    dim = len(modes)

    if opts is None:
        opts = default_opts(finufft_type, dim)

    modes = modes + (1,) * (3 - dim)
    modes = (c_int * 3)(*modes)

    plan = CufinufftPlan()
    plan.opts = opts

    ier = _make_plan(finufft_type - 1, dim, modes, isign, 1, float(tol), 1, plan)

    if ier != 0:
        raise RuntimeError('Error creating plan.')

    return plan

_set_nu_pts = lib.cufinufftc_setNUpts
_set_nu_pts.argtypes = [
    c_int, c_void_p, c_void_p, c_void_p, ctypes.c_int, c_float_p,
    c_float_p, c_float_p, CufinufftPlan_p]
_set_nu_pts.restype = c_int

def set_nu_pts(plan, M, kx, ky=None, kz=None):
    if ky is None: ky = 0
    if kz is None: kz = 0

    ier = _set_nu_pts(M, int(kx), int(ky), int(kz), 0, None, None, None, plan)

    if ier != 0:
        raise RuntimeError('Error setting non-uniform points.')

_exec_plan = lib.cufinufftc_exec
_exec_plan.argtypes = [c_void_p, c_void_p, CufinufftPlan_p]
_exec_plan.restype = c_int

def execute(plan, c, fk):
    ier = _exec_plan(int(c), int(fk), plan)

    if ier != 0:
        raise RuntimeError('Error executing plan.')

_destroy_plan = lib.cufinufftc_destroy
_destroy_plan.argtypes = [CufinufftPlan_p]
_destroy_plan.restype = c_int

def destroy(plan):
    ier = _destroy_plan(plan)

    if ier != 0:
        raise RuntimeError('Error destroying plan.')


__all__ = ['CufinufftPlan', 'NufftOpts', 'SpreadOpts', 'default_opts',
        'make_plan', 'set_nu_pts', 'exec_plan', 'destroy_plan']
