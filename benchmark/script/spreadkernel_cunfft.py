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
	reps=10
	density = 1.0
	N_totry = 2**np.arange(8,13)
	t_cunfft_conv = np.zeros(len(N_totry))
	for i,N in enumerate(N_totry):
		M = int((N/2.0)*(N/2.0))
		# Method 1
		t = 0
		for n in range(reps):
                        output=subprocess.check_output(["./cunfft_timing",str(N),str(N),str(M)], \
                                            cwd="./").decode("utf-8")
                        t+= float(find_between(output, "cnufft spread", "ms"))
		t_cunfft_conv[i] = t/reps
	
	# Output result
	print("cunfft_timing: kernel conv")
	for i,N in enumerate(N_totry):
		print('N={:5d}, t= {:5.3e}'.format(N,t_cunfft_conv[i]))
  
if __name__== "__main__":
  main()
