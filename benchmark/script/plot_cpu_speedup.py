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
	num_lines = sum(1 for line in open('../results/cputime_nonuniform_nupts_0730.out', 'r'))
	N = np.zeros(num_lines, dtype=int)
	s_cpu = np.zeros(num_lines)
	s_gpu = np.zeros([4, num_lines])
	speedup = np.zeros([4, num_lines])
	f = open('../results/cputime_nonuniform_nupts_0730.out', 'r')
	for i,line in enumerate(f):
        	N[i]=int(find_between(line, 'N=', ','))
		s_cpu[i]=float(find_between(line, 's=', '\n'))
	f = open('../results/gputime_nonuniform_nupts_0802.out', 'r')
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
	fig, ax1 = plt.subplots(figsize=(10, 5))
	ax2 = ax1.twinx()
	w = 0.05
	x = np.array(x)
	ax2.plot(x+1*w, speedup[0,:],'-o',label='input driven without sort')
	ax2.plot(x+2*w, speedup[1,:],'-o', label='input driven with sort')
	ax2.plot(x+3*w, speedup[2,:],'-o', label='hybrid')
	ax2.plot(x+4*w, speedup[3,:],'-o', label='subprob')
	ax2.axhline(y=1, linestyle='--', color='k',label='cpu')
	ax2.set_ylim((0, 10))
	ax2.set_ylabel('speedup')
	leg = ax2.legend(loc=0,frameon=1)
	leg.get_frame().set_alpha(0.5)
	
	opacity=0.8
	ax1.bar(x+0.5*w,s_gpu[0,:],w,alpha=opacity,label='input driven without sort')
	ax1.bar(x+1.5*w,s_gpu[1,:],w,alpha=opacity,label='input driven with sort')
	ax1.bar(x+2.5*w,s_gpu[2,:],w,alpha=opacity,label='hybrid')
	ax1.bar(x+3.5*w,s_gpu[3,:],w,alpha=opacity,label='subprob')
	ax1.bar(x-0.5*w,s_cpu,w,alpha=opacity,label='cpu')
	ax1.set_xticks(x+2.5*w)
    	ax1.set_xticklabels(N)
	ax1.set_xlim((x[0]-0.5, x[-1]+0.5))
	ax1.set_xlabel('N')
	ax1.set_ylabel('#NU pts/s')
	ax1.set_title('T_cpuspreader/T_gpuspreader')
	plt.grid(True)

	plt.savefig('../speedup_nonuniform_nupts_cnufftspread_vs_gpuspread_0802.pdf')
	plt.show()
if __name__== "__main__":
  main()
