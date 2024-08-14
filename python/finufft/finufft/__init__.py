"""The Python interface to FINUFFT is divided into two parts: the simple
interface (through the ``nufft*`` functions) and the more advanced plan
interface (through the ``Plan`` class). The former allows the user to perform
an NUFFT in a single call while the latter allows for more efficient reuse of
resources when the same NUFFT is applied several times to different data by
saving FFTW plans, sorting the nonuniform points, and so on.
"""

# that was the docstring for the package finufft.

__all__ = ["nufft1d1","nufft1d2","nufft1d3","nufft2d1","nufft2d2","nufft2d3","nufft3d1","nufft3d2","nufft3d3","Plan"]
# etc..

# let's just get guru and nufft1d1 working first...
from finufft._interfaces import Plan
from finufft._interfaces import nufft1d1,nufft1d2,nufft1d3
from finufft._interfaces import nufft2d1,nufft2d2,nufft2d3
from finufft._interfaces import nufft3d1,nufft3d2,nufft3d3

__version__ = '2.3.0-rc1'
