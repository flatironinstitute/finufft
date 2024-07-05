Julia interfaces (CPU and GPU)
==============================

Principal author Ludvig af Klinteberg and others have built and maintain `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_, an interface from the `Julia <https://julialang.org/>`_ language. This official Julia package supports 32-bit and 64-bit precision, now on both CPU and GPU (via `CUDA.jl`), via a common interface.
The Julia package installation automatically downloads pre-built CPU binaries of the FINUFFT library for Linux, macOS, Windows and FreeBSD (for a full list see `finufft_jll <https://github.com/JuliaBinaryWrappers/finufft_jll.jl>`_), and the GPU binary for Linux (see `cufinufft_jll <https://github.com/JuliaBinaryWrappers/cufinufft_jll.jl>`_).

`FINUFFT.jl` has itself been wrapped as part of `NFFT.jl <https://juliamath.github.io/NFFT.jl/dev/performance/>`_, which contains an "abstract" interface
to any NUFFT in Julia, with FINUFFT as an example.
Their
`performance comparison page <https://juliamath.github.io/NFFT.jl/dev/performance/>`_
show that FINUFFT matches their native Julia implementation for speed of type 1
and type 2 transforms
in 3D, and beats NFFT, and with less precomputation.
In 1D and 2D, the native Julia implementation is 1-2 times faster
than FINUFFT in their tests on uniformly-random nonuniform points.
