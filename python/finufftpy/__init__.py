"""Python wrappers to FINUFFT (Flatiron Institute nonuniform fast Fourier transform library).

*** Alex: this all needs to be fixed ***

Authors: Daniel Foreman-Mackey, Jeremy Magland, and Alex Barnett

Contents:

(out of date)

nufft1d1
nufft1d2
nufft1d3
nufft2d1
nufft2d1many
nufft2d2
nufft2d2many
nufft2d3
nufft3d1
nufft3d2
nufft3d3

"""

# that was the docstring for the package finufftpy.

__all__ = ["finufft_plan","default_opts","nufft_opts","destroy","makeplan","setpts","execute","nufft1d1"]
# etc..

# alex: why is finufftpy_cpp module needed?
from finufftpy_cpp import default_opts
from finufftpy_cpp import destroy
from finufftpy_cpp import nufft_opts
from finufftpy_cpp import finufft_plan

# let's just get guru and nufft1d1 working first...
from finufftpy._interfaces import makeplan,setpts,execute,nufft1d1
