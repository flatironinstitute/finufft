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
	num_lines = sum(1 for line in open('../results/compare_cunfft_0731.out', 'r'))-1
	N = np.zeros(num_lines, dtype=int)
	t_cunfft = np.zeros(num_lines)
	t_nfft = np.zeros(num_lines)
	t_cufinufft = np.zeros([3, num_lines])
	t_finufft = np.zeros(num_lines)
	#speedup = np.zeros([4, num_lines])
	f = open('../results/compare_cunfft_0731.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 'N=', ',')
		if(temp==""):
                        continue
		else:
        		N[i]= int(temp)
			t_cunfft[i]=float(find_between(line, 't=', '\n'))
			i=i+1

	f = open('../results/compare_cufinufft_0731.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_cufinufft[i/num_lines, i%num_lines]=float(temp)
			i=i+1

	f = open('../results/compare_finufft_0731.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_finufft[i]=float(temp)
			i=i+1

	f = open('../results/compare_nfft_0731.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_nfft[i]=float(temp)
			i=i+1

	#for m in range(4):
	#	speedup[m,:] = np.divide(t_cunfft, t_finufft[m,:])
	x = 0.0
	w = 0.01
	fig, ax= plt.subplots(2,3,figsize=(30, 10))
	for nn in range(len(N)):
		i=nn/3
		j=nn%3
		ax[i,j].bar(x, t_nfft[nn], w, color='darkblue', log=1, label='nfft')
		ax[i,j].bar(x+w, t_finufft[nn], w, color='slateblue',log=1, label='finufft')
		ax[i,j].bar(x+2*w, t_cunfft[nn], w, color='blueviolet',log=1, label='cunfft')
		ax[i,j].bar(x+3*w, t_cufinufft[0,nn], w, color='darkviolet',log=1, label='input driven without sort')
		ax[i,j].bar(x+4*w, t_cufinufft[1,nn], w, color='violet',log=1, label='input driven with sort')
		ax[i,j].bar(x+5*w, t_cufinufft[2,nn], w, color='deeppink',log=1, label='hybrid')

		ax[i,j].set_xticks([x+3*w])
		ax[i,j].set_xticklabels(N[nn:nn+1])
		ax[i,j].set_xlim([x-0.5*w, x+6.5*w])
		ax[i,j].set_ylabel('ms')
		ax[i,j].set_xlabel('N')
		ax[i,j].grid()
		#ax[i,j].set_ylim((0, 12))
		#ax[i,j].set_title('T_cunfft/T_cufinufft')
	
	handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6)
	plt.savefig('../time_all.pdf')
	plt.show()
if __name__== "__main__":
  main()
