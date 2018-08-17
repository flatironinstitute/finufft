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
	t_cufinufft_s_1 = np.load(result_dir+"cufinufft_spread_s_1_0816.npy")
	t_cunfft_s_1    = np.load(result_dir+"cunfft_spread_s_1_0812.npy")
	t_finufft_s_1   = np.load(result_dir+"finufft_spread_s_1_0812.npy")

	density = 10.0**np.arange(-1,2)#0.1, 1, 10
	tol     = 10.0**np.linspace(-14,-2,4) #1e-14, 1e-10, 1e-6 , 1e-2
	nf1     = 2**np.arange(7,13) #128, ... ,4096

	d = 1
	t = 2
	M = nf1/2*nf1/2*density[d]*10**3
	fig, ax= plt.subplots(1,1,figsize=(10, 6))

	ax.axhline(y=1, linestyle=':', color='k')
	ax.set_title('Speedup (t_double/t_single)')
	x = np.array(range(len(nf1)))
	ax.plot(x, t_finufft_d_1[d,t,:]/t_finufft_s_1[0,t,:],'-o',label='FINUFFT')
	ax.plot(x, t_cunfft_d_1[d,t,:]/t_cunfft_s_1[0,t,:],'-o',label='cuNFFT')
	ax.plot(x, t_cufinufft_d_1[d,t,:]/t_cufinufft_s_1[0,t,:],'-o',label='cuFINUFFT')
    
	ax.set_xlabel('nf1=nf2')
	ax.set_xticks(x)
	ax.set_xticklabels(nf1)
	ax.legend(loc=0)
	ax.grid()

	fig.suptitle('Uniform distributed pts, Double Precision, density='+str(density[d]), fontsize=15)
	plt.show()
	#fig.savefig('d_u_1e-6_den1.eps')
if __name__== "__main__":
	main()
