import finufftpy_cpp
import numpy as np

__all__ = [
    "finufft1d1",
]

def finufft1d1(xj,cj,iflag,eps,ms,fk):
	# fk is the output and must have dtype=np.complex128
	xj=xj.astype(np.float64,copy=False) #copies only if type changes
	cj=cj.astype(np.complex128,copy=False) #copies only if type changes
	return finufftpy_cpp.finufft1d1_cpp(xj,cj,iflag,eps,ms,fk)