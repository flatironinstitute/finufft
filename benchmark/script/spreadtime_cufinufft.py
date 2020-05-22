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
	t_gpuspread_5 = np.zeros([len(density_totry), len(tol_totry), len(nf1_totry)])
	f= open("../results/cufinufft_spread_s_1_0816.out","w")
	f.write("cufinufft spread: method(5) subprob\n")
	f.write('(density,tol,nf1,M)\tTime(HtoD(ms) + Spread(ms) + DtoH(ms))\n')
	for d,density in enumerate(density_totry):
		for t,tol in enumerate(tol_totry):
			for n,nf1 in enumerate(nf1_totry):
				M = int((nf1/2.0)*(nf1/2.0)*density)
				# Method 1
				"""
				t = 0
				for n in range(reps):
					output=subprocess.check_output(["bin/spread2d",'1',str(nupts_distr),str(N),str(N)], \
							    cwd="../../").decode("utf-8")
					t+= float(find_between(output, "HtoD", "ms"))
					t+= float(find_between(output, "Spread", "ms"))
					t+= float(find_between(output, "DtoH", "ms"))
				t_gpuspread_1[i] = t/reps
				"""
				# Method 5
				print(d,t,n)
				tnow = float('Inf')
				for nn in range(reps):
					tt = 0.0
					output=subprocess.check_output(["bin/spread2d",'5',str(nupts_distr),str(nf1),str(nf1),str(M), \
									 str(tol)], cwd="../../").decode("utf-8")
					tt+= float(find_between(output, "HtoD", "ms"))
					tt+= float(find_between(output, "Spread (5)", "ms"))
					tt+= float(find_between(output, "DtoH", "ms"))
					tnow = min(tnow,tt)
				t_gpuspread_5[d,t,n] = tnow
				f.write('({:5.1e},{:5.1e},{:5d},{:15d})\t t={:5.3e}\n'.format(density,tol,nf1,M,t_gpuspread_5[d,t,n]))
	
	np.save('../results/cufinufft_spread_s_1_0816.npy', t_gpuspread_5)

	# Output result
	"""
	print("Method 1: input driven without sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuspread_1[i]))
	print("\n")

	print("Method 2: input driven with sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuspread_2[i]))
	print("\n")

	print("Method 4: hybrid")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuspread_4[i]))
	print("\n")
	print("Method 5: subprob")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_gpuspread_5[i]))
  	"""
if __name__== "__main__":
  main()
