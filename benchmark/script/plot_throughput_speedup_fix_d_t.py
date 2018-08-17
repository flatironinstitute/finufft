import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

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
	t = 2
	M = nf1/2*nf1/2*density[d]*10**3
	fig, ax= plt.subplots(1,2,figsize=(15, 6))

	w = 0.2
	x = np.array(range(len(nf1)))

	ax[0].bar(x, M/t_nfft_d_1[d,0,:],w,color="darkblue",label='NFFT (24 thread)')
	ax[0].bar(x+w, M/t_finufft_d_1[d,t,:],w, color="darkviolet",label='FINUFFT (24 threads)')
	ax[0].bar(x+2*w, M/t_cunfft_d_1[d,t,:],w, color="violet",label='cuNFFT')
	ax[0].bar(x+3*w, M/t_cufinufft_d_1[d,t,:],w, color="deeppink",label='cuFINUFFT')
	formatter = ticker.ScalarFormatter()
	formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))

	ax[0].set_xlabel('nf1=nf2')
	ax[0].set_title('Throughput (#NU pts/s)')
	ax[0].set_xticks(x+1.5*w)
	ax[0].set_xticklabels(nf1)
	ax[0].yaxis.set_major_formatter(formatter)
	ax[0].legend(loc=0)
	ax[0].grid()

	ax[1].axhline(y=1, linestyle='--', color='k')
	ax[1].set_title('Speed up (s/s_FINUFFT)')
	ax[1].plot(x, t_finufft_d_1[d,t,:]/t_nfft_d_1[d,0,:],'-o',color='darkblue')
	ax[1].plot(x, t_finufft_d_1[d,t,:]/t_cunfft_d_1[d,t,:],'-o',color='violet')
	ax[1].plot(x, t_finufft_d_1[d,t,:]/t_cufinufft_d_1[d,t,:],'-o',color='deeppink')
	ax[1].set_xticks(x)
	ax[1].set_xticklabels(nf1)
	ax[1].set_xlabel('nf1=nf2')
	ax[1].set_ylim(bottom=0)
	ax[1].grid()
	fig.suptitle('Uniform distributed pts, Double Precision, tol='+str(tol[t])+'(ns ='+str(-np.log10(density[d])+1)+'), density='+str(density[d]), fontsize=15)
	plt.show()
	#fig.savefig('d_u_1e-6_den1.eps')
if __name__== "__main__":
	main()
