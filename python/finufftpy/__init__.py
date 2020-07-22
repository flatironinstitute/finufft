"""Python wrappers to FINUFFT (Flatiron Institute nonuniform fast Fourier transform library).

*** Alex: this all needs to be fixed ***

Authors: Daniel Foreman-Mackey, Jeremy Magland, and Alex Barnett

Contents:

(out of date)

Plan
nufft1d1
nufft1d2
nufft1d3
nufft2d1
nufft2d2
nufft2d3
nufft3d1
nufft3d2
nufft3d3

"""

# that was the docstring for the package finufftpy.

__all__ = ["Plan","nufft1d1","nufft1d2","nufft1d3","nufft2d1","nufft2d2","nufft2d3","nufft3d1","nufft3d2","nufft3d3"]
# etc..

# let's just get guru and nufft1d1 working first...
from finufftpy._interfaces import Plan
from finufftpy._interfaces import nufft1d1,nufft1d2,nufft1d3
from finufftpy._interfaces import nufft2d1,nufft2d2,nufft2d3
from finufftpy._interfaces import nufft3d1,nufft3d2,nufft3d3
