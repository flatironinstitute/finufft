import subprocess
import numpy as np
import re

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def main():
	nupts_distr=2
	reps=5
	density_totry = 10.0**np.arange(-1,2) #0.01, 0.1, 1, 10, 100
	#density_totry = 10.0**np.arange(0,1) #0.01, 0.1, 1, 10, 100
	tol_totry     = 10.0**np.linspace(-14,-2,4) #1e-14, 1e-10, 1e-6 , 1e-2
	nf1_totry     = 2**np.arange(7,13) #128, ... ,4096 
	t_gpuspread_5 = np.zeros([len(density_totry), len(tol_totry), len(nf1_totry)])
	f= open("../results/cunfft_interp_d_2_0812.out","w")
	f.write("cunfft interp (clustered nupts):\n")
	f.write('(density,tol,nf1,M)\tTime(HtoD(ms) + Spread(ms) + DtoH(ms))\n')
	for t,tol in enumerate(tol_totry):
		cutoff=int((-np.log10(tol/10.0))/2.0)
		print("cutoff={:5f}".format(int(cutoff)))
		subprocess.call(["ls"],cwd="../../../CUNFFT/build/")
		subprocess.call(["make", "distclean"],cwd="../../../CUNFFT/build/")
		subprocess.call(["cmake", "..",  "-DCUT_OFF="+str(cutoff), "-DCUNFFT_DOUBLE_PRECISION=ON", "-DMILLI_SEC=ON", "-DMEASURED_TIMES=ON", \
                                         "-DCUDA_CUDA_LIBRARY=/usr/lib64/nvidia/libcuda.so", "-DPRINT_CONFIG=OFF"],cwd="../../../CUNFFT/build/")
		subprocess.call(["make"],cwd="../../../CUNFFT/build/") #> makeLog.txt 2>&1
		subprocess.call(["make", "install"], cwd="../../../CUNFFT/build/")
		for d,density in enumerate(density_totry):
			for n,nf1 in enumerate(nf1_totry):
				M = int((nf1/2.0)*(nf1/2.0)*density)
				# Method 5
				tnow = float('Inf')
				print(t,d,n)
				for nn in range(reps):
					tt = 0.0
					output=subprocess.check_output(["./cunfft_timing",str(nupts_distr),str(nf1),str(nf1),str(M)], cwd="../").decode("utf-8")
					tt+= float(find_between(output, "HtoD", "ms"))
					tt+= float(find_between(output, "interp", "ms"))
					tt+= float(find_between(output, "DtoH", "ms"))
					tnow = min(tnow,tt)
				t_gpuspread_5[d,t,n] = tnow
				f.write('({:5.1e},{:5.1e},{:5d},{:15d})\t t={:5.3e}\n'.format(density,tol,nf1,M,t_gpuspread_5[d,t,n]))
	np.save('../results/cunfft_interp_d_2_0812.npy', t_gpuspread_5)

if __name__== "__main__":
  main()
