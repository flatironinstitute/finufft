import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def main():
	num_lines = sum(1 for line in open('../results/cputime.out', 'r'))
	N = np.zeros(num_lines, dtype=int)
	s_cpu = np.zeros(num_lines)
	s_gpu = np.zeros([4, num_lines])
	speedup = np.zeros([4, num_lines])
	f = open('../results/cputime.out', 'r')
	for i,line in enumerate(f):
        	N[i]=int(find_between(line, 'N=', ','))
		s_cpu[i]=float(find_between(line, 's=', '\n'))
	f = open('../results/gputime.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 's=', '\n')
		if(temp==""):
			continue
		else:
			s_gpu[i/num_lines, i%num_lines]=float(temp)
			i=i+1
	for m in range(4):
		speedup[m,:] = np.divide(s_gpu[m,:],s_cpu)
	x = range(len(N))
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(x, speedup[0,:],'-o', label='input driven without sort')
	ax.plot(x, speedup[1,:],'-o', label='input driven with sort')
	ax.plot(x, speedup[2,:],'-o', label='output driven')
	ax.plot(x, speedup[3,:],'-o', label='hybrid')
	ax.plot(x, np.ones(len(N)), '--', label='cpu')
	ax.set_xticks(x)
    	ax.set_xticklabels(N)
	ax.set_xlim((x[0]-0.5, x[-1]+0.5))
	ax.set_xlabel('N')
	ax.set_ylabel('speedup')
	ax.set_title('T_cpuspreader/T_gpuspreader')
	leg = ax.legend(loc=0,frameon=1)
	leg.get_frame().set_alpha(0.5)
	plt.grid(True)
	plt.savefig('../speedup_cnufftspread_vs_gpuspraed.pdf')
if __name__== "__main__":
  main()
