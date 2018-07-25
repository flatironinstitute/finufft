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
	num_lines = sum(1 for line in open('compare_cunfft_0725.out', 'r'))-1
	N = np.zeros(num_lines, dtype=int)
	t_cunfft = np.zeros(num_lines)
	t_finufft = np.zeros([4, num_lines])
	speedup = np.zeros([4, num_lines])
	f = open('compare_cunfft_0725.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 'N=', ',')
		if(temp==""):
                        continue
		else:
        		N[i]= int(temp)
			t_cunfft[i]=float(find_between(line, 't=', '\n'))
			i=i+1
	f = open('compare_finufft_0725.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 's=', '\n')
		if(temp==""):
			continue
		else:
			t_finufft[i/num_lines, i%num_lines]=float(temp)
			i=i+1
	for m in range(4):
		speedup[m,:] = np.divide(t_cunfft, t_finufft[m,:])
	x = range(len(N))
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(x, speedup[0,:],'-o', label='input driven without sort')
	ax.plot(x, speedup[1,:],'-o', label='input driven with sort')
	ax.plot(x, speedup[2,:],'-o', label='output driven')
	ax.plot(x, speedup[3,:],'-o', label='hybrid')
	ax.plot(x, np.ones(len(N)), '--', label='cunfft')
	ax.set_xticks(x)
    	ax.set_xticklabels(N)
	ax.set_xlim((x[0]-0.5, x[-1]+0.5))
	ax.set_ylim((0, 12))
	ax.set_xlabel('N')
	ax.set_ylabel('speedup')
	ax.set_title('T_cunfft/T_finufft')
	leg = ax.legend(loc=0,frameon=1)
	leg.get_frame().set_alpha(0.5)
	plt.grid(True)
	plt.show()
	plt.savefig('speedup_cunfft_vs_finufft.pdf')
if __name__== "__main__":
  main()
