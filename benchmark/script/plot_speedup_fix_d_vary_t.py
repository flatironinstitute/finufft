import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def main():
	result_dir="../results/"
	t_cufinufft_d_1 = np.load(result_dir+"cufinufft_spread_d_1_0812.npy")
	t_finufft_d_1   = np.load(result_dir+"finufft_spread_d_1_0812.npy")
	t_cunfft_d_1    = np.load(result_dir+"cunfft_spread_d_1_0812.npy")
	t_nfft_d_1      = np.load(result_dir+"nfft_spread_d_1_0816.npy")

	density = 10.0**np.arange(-1,2)#0.1, 1, 10
	tol     = 10.0**np.linspace(-14,-2,4) #1e-14, 1e-10, 1e-6 , 1e-2
	nf1     = 2**np.arange(7,13) #128, ... ,4096

	d = 1
	M = nf1/2*nf1/2*density[d]*10**3
	w = 0.1
	fig, ax= plt.subplots(1,2,figsize=(15, 6))
	ax[0].axhline(y=1, linestyle='--', color='k')
	ax[0].set_title('Speedup (s_cuFINUFFT/s_FINUFFT)')
	ax[1].axhline(y=1, linestyle='--', color='k')
	ax[1].set_title('Speedup (s_cuFINUFFT/s_cuNFFT)')
	x = np.array(range(len(tol)))
	for n in range(len(nf1)):
    		ax[0].plot(x*w+n,flip(t_finufft_d_1[d,:,n]/t_cufinufft_d_1[d,:,n],0),'-o')
    		ax[1].plot(x*w+n,flip(t_cunfft_d_1[d,:,n] /t_cufinufft_d_1[d,:,n],0),'-o')
	ax[0].set_ylim(bottom=0)
	ax[1].set_ylim(bottom=0)

	x = np.array(range(len(nf1)))
	ax[0].set_xticks(x)
	ax[0].set_xticklabels(nf1)
	ax[0].set_xlabel('nf1=nf2')
	ax[1].set_xticks(x)
	ax[1].set_xticklabels(nf1)
	ax[1].set_xlabel('nf1=nf2')
	ax[0].grid()
	ax[1].grid()

	fig.suptitle('Uniform distributed pts, Double Precision, density='+str(density[d]), fontsize=15)
	plt.show()
	#fig.savefig('d_u_1e-6_den1.eps')
if __name__== "__main__":
	main()
