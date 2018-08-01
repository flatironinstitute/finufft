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
	reps=100
	density = 1.0
	N_totry = 2**np.arange(8,13)
	s_cpuspread = np.zeros(len(N_totry))
	for i,N in enumerate(N_totry):
		t = 0
		M = int((N/2.0)*(N/2.0))
		for n in range(reps):
			output=subprocess.check_output(["./spreadtestnd_mel",'2',str(M),str(N*N),'1e-6'], \
                                            cwd="../").decode("utf-8")
			s+= float(find_between(output, "=>", "pts/s"))
		s_cpuspread[i] = s/reps
		print('N={:5d}, t= {:5.3e}'.format(N,s_cpuspread[i]))
  
if __name__== "__main__":
  main()
