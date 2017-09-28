import numpy as np
import finufftpy

nj=1000
ms=100
iflag=1
eps=1e-6
xj=np.random.rand(1,nj)
cj=np.random.rand(1,nj)+1j*np.random.rand(1,nj);
fk=np.zeros([1,ms],dtype=np.complex128)
ret=finufftpy.finufft1d1(xj,cj,iflag,eps,ms,fk)
print(ret)
print(fk.shape)
for j in range(0,10):
	print(j,fk[0][j])



