"""
Demonstrate the 3D type 2 NUFFT using cuFINUFFT with overlapping transfer/compute and
compare to other methods
"""

import numpy as np
import cupy
import cufinufft
import time

rng = np.random.Generator(np.random.PCG64(100))


pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def test():
    # Set up parameters for problem.
    N = (128, 128, 160)             # Size of uniform grid
    M = 6136781                     # Number of nonuniform points
    n_transf = 1                    # Number of nuffts to perform per batch
    n_tot = 32                      # Number of batches to execute
    eps = 1e-5                      # Requested tolerance
    dtype = np.float32              # Datatype (real)
    complex_dtype = np.complex64    # Datatype (complex)

    # Generate coordinates of non-uniform points.
    x = 2 * np.pi * (rng.random(size=M, dtype=dtype) - 0.5)
    y = 2 * np.pi * (rng.random(size=M, dtype=dtype) - 0.5)
    z = 2 * np.pi * (rng.random(size=M, dtype=dtype) - 0.5)

    # Generate grid values.
    fk_all_naive = (rng.standard_normal((n_tot, n_transf, *N), dtype=dtype) + 1j * rng.standard_normal((n_tot, n_transf, *N), dtype=dtype))
    fk_all = _pin_memory(fk_all_naive)
    c_all_async = _pin_memory(np.zeros(shape=(n_tot, n_transf, M), dtype=complex_dtype))
    c_all_sync = _pin_memory(np.zeros(shape=(n_tot, n_transf, M), dtype=complex_dtype))
    c_all_naive = np.zeros(shape=(n_tot, n_transf, M), dtype=complex_dtype)

    # Initialize the plan and set the points.
    plan_stream = cupy.cuda.Stream(null=True)
    plan = cufinufft.Plan(2, N, n_transf, eps=eps, dtype=complex_dtype, gpu_kerevalmeth=1, gpu_stream=plan_stream.ptr)
    plan.setpts(cupy.array(x), cupy.array(y), cupy.array(z))

    # Using a simple front/back buffer approach. backbuffer is for DtoH transfers, and front for
    # execution and HtoD transfers
    front_stream, back_stream = (cupy.cuda.Stream(), cupy.cuda.Stream())
    front_fk_gpu = cupy.empty(fk_all[0].shape, fk_all[0].dtype)
    back_fk_gpu = cupy.empty(fk_all[0].shape, fk_all[0].dtype)
    front_c_gpu = cupy.empty(c_all_async[0].shape, c_all_async[0].dtype)
    back_c_gpu = cupy.empty(c_all_async[0].shape, c_all_async[0].dtype)
    front_plan = cufinufft.Plan(2, N, n_transf, eps=eps, dtype=complex_dtype, gpu_kerevalmeth=1, gpu_stream=front_stream.ptr)
    front_plan.setpts(cupy.array(x), cupy.array(y), cupy.array(z))
    back_plan = cufinufft.Plan(2, N, n_transf, eps=eps, dtype=complex_dtype, gpu_kerevalmeth=1, gpu_stream=back_stream.ptr)
    back_plan.setpts(cupy.array(x), cupy.array(y), cupy.array(z))

    # Run with async
    st = time.time()
    front_fk_gpu.set(fk_all[0], stream=front_stream)
    for i in range(n_tot):
        if i + 1 < n_tot:
            back_fk_gpu.set(fk_all[i + 1], stream=back_stream)

        front_plan.execute(front_fk_gpu, out=front_c_gpu).get(out=c_all_async[i], stream=front_stream)

        back_stream, front_stream = front_stream, back_stream
        back_plan, front_plan = front_plan, back_plan
        back_fk_gpu, front_fk_gpu = front_fk_gpu, back_fk_gpu
        back_c_gpu, front_c_gpu = front_c_gpu, back_c_gpu

    back_stream.synchronize()
    async_time = time.time() - st

    # Run with best-practice synchronous
    st = time.time()
    for i in range(n_tot):
        front_fk_gpu.set(fk_all[i])
        plan.execute(front_fk_gpu, out=front_c_gpu).get(out=c_all_sync[i])

    sync_time = time.time() - st

    # Run in a relatively naive way
    st = time.time()
    for i in range(n_tot):
        c_all_naive[i, :] = plan.execute(cupy.array(fk_all_naive[i])).get()[:]

    naive_time = time.time() - st

    assert(np.linalg.norm(c_all_sync - c_all_naive) == 0.0)
    assert(np.linalg.norm(c_all_async - c_all_naive) == 0.0)

    print(f"async timing: {async_time}")
    print(f"sync timing: {sync_time}")
    print(f"naive timing: {naive_time}")
    print(f"speedup (sync / async): {round(sync_time / async_time, 2)}")
    print(f"speedup (naive / sync): {round(naive_time / sync_time, 2)}")
    print(f"speedup (naive / async): {round(naive_time / async_time, 2)}")

    # Since plans carry raw stream pointers which aren't reference counted, we need to make
    # sure they're deleted before the stream objects that hold them. Otherwise, the stream
    # might be deleted before cufinufft can use it in the deletion routines. Manually clear
    # them out here.
    del plan, front_plan, back_plan

test()
