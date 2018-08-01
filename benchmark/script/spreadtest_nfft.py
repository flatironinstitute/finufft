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
	N_totry = 2**np.arange(7,13)
	t_nfft = np.zeros(len(N_totry))
	for i,N in enumerate(N_totry):
		t = 0
		M = int((N/2.0)*(N/2.0))
		N1 = int(N/2.0)
		N2 = int(N/2.0)
		for n in range(reps):
			output=subprocess.check_output(["./nfft_simpletest",str(N1),str(N2),str(M)], \
                                            cwd="../").decode("utf-8")
			t+= float(find_between(output, "Spread", "ms"))
		t_nfft[i] = t/reps
		print('N={:5d}, t= {:5.3e}'.format(N,t_nfft[i]))
  
if __name__== "__main__":
  main()
