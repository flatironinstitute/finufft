Julia interfaces
================

Ludvig af Klinteberg, Libin Lu, and others, have built `FINUFFT.jl <https://github.com/ludvigak/FINUFFT.jl>`_, an interface from the `Julia <https://julialang.org/>`_ language. This package supports 32-bit and 64-bit precision, and automatically downloads and runs pre-built binaries of the FINUFFT library for Linux, macOS, Windows and FreeBSD (for a full list see `finufft_jll <https://github.com/JuliaBinaryWrappers/finufft_jll.jl>`_).

`FINUFFT.jl` has now (in 2022) itself been wrapped as part of `NFFT.jl <https://juliamath.github.io/NFFT.jl/dev/performance/>`_, which contains an "abstract" interface
to any NUFFT in Julia, with FINUFFT as an example.
Their
`performance comparison page <https://juliamath.github.io/NFFT.jl/dev/performance/>`_
show that FINUFFT matches their native Julia implementation for speed of type 1
and type 2 transforms
in 3D, and beats NFFT, and with less precomputation.
In 1D and 2D, the native Julia implementation is 1-2 times faster
than FINUFFT in their tests on uniformly-random nonuniform points.
