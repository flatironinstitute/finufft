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
	M = nf1/2*nf1/2*density[d]*10**3
	fig, ax= plt.subplots(1,1,figsize=(10, 8))

	color = ["darkblue", "slateblue", "darkviolet", "violet"]
	#color = ["plum", "lightpink", "mediumvioletred", "palevioletred"]
	w = 0.1
	x = np.array(range(len(nf1)))
	for t in range(len(tol)):
    		ns = -np.log10(tol[-t-1]/10.0)
    		ax.bar(x+t*w, M/t_cunfft_d_1[d,-(t+1),:],w,color=color[t],label=str(tol[-(t+1)])+" (w="+str(int(ns))+")")
		formatter = ticker.ScalarFormatter()
		formatter.set_scientific(True)
	formatter.set_powerlimits((-1,1))
    
	ax.set_xlabel('nf1=nf2')
	ax.set_title('Throughput (#NU pts/s)')
	ax.set_xticks(x+2*w)
	ax.set_xticklabels(nf1)
	ax.yaxis.set_major_formatter(formatter)
	ax.legend(loc=0)
	ax.grid()

	fig.suptitle('Uniform distributed pts, Double Precision, density='+str(density[d]), fontsize=15)
	plt.show()
	#fig.savefig('d_u_1e-6_den1.eps')
if __name__== "__main__":
	main()
