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
	nupts_distr=1
	reps=5
	#density_totry = 10.0**np.arange(-1,2) #0.1, 1, 10
	density_totry = 10.0**np.arange(0,1) #0.1, 1, 10
	tol_totry     = 10.0**np.linspace(-14,-2,4) #1e-14, 1e-10, 1e-6 , 1e-2
	nf1_totry     = 2**np.arange(7,13) #128, ... ,4096 
	t_gpuinterp_5 = np.zeros([len(density_totry), len(tol_totry), len(nf1_totry)])
	f= open("../results/nfft_interp_d_1_0812.out","w")
	f.write("nfft interp:\n")
	f.write('(density,tol,nf1,M)\tTime(HtoD(ms) + Spread(ms) + DtoH(ms))\n')
	for d,density in enumerate(density_totry):
		for t,tol in enumerate(tol_totry):
			for n,nf1 in enumerate(nf1_totry):
				M = int((nf1/2.0)*(nf1/2.0)*density)
				# Method 5
				print(d,t,n)
				tnow = float('Inf')
				for nn in range(reps):
					tt = 0.0
					output=subprocess.check_output(["./nfft_simpletest",str(nupts_distr),str(nf1/2),str(nf1/2),str(M), \
									 str(tol)], cwd="../").decode("utf-8")
					tt+= float(find_between(output, "Interp", "ms"))
					tnow = min(tnow,tt)
				t_gpuinterp_5[d,t,n] = tnow
				f.write('({:5.1e},{:5.1e},{:5d},{:15d})\t t={:5.3e}\n'.format(density,tol,nf1,M,t_gpuinterp_5[d,t,n]))
	
	np.save('../results/nfft_interp_d_1_0812.npy', t_gpuinterp_5)

	# Output result
	"""
	print("Method 1: input driven without sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuinterp_1[i]))
	print("\n")

	print("Method 2: input driven with sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuinterp_2[i]))
	print("\n")

	print("Method 4: hybrid")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuinterp_4[i]))
	print("\n")
	print("Method 5: subprob")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuinterp_5[i]))
  	"""
if __name__== "__main__":
  main()
