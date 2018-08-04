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
	num_n=0
	num_method=0
	for line in open('../results/gputime_kernel_breakdown_0803_2.out', 'r'):
		temp=find_between(line, 'N=', '\n')
		if(temp==""):
			temp=find_between(line, 'Method', '\n')
			if(temp==""):
				continue
			else:
				num_method+=1
		else:
			num_n+=1
	num_n = num_n/num_method
	t_methods = np.zeros([num_n, num_method, 8])
	N = np.zeros(num_n, dtype=int)
	
	f = open('../results/gputime_kernel_breakdown_0803_2.out', 'r')
	j=0
	for line in f:
		temp = find_between(line, 'N=', '\n')
		if(temp==""):
			if(find_between(line, 'Method 2', '\n') != ""):
				break
		else:
			N[j] = int(temp)
			j+=1

	f = open('../results/gputime_kernel_breakdown_0803_2.out', 'r')
	m = -1
	n = -1
	for i,line in enumerate(f):
		if(find_between(line, 'Method', '\n')!=""):
			m=m+1
			continue
		if(find_between(line, 'N=', '\n')!=""):
			n=n+1
			if(n==num_n):
				n=0
			continue
		else:
			k = find_between(line, 'k=', '\t')
			t=find_between(line,'\t', '\n')
			t_methods[n,m,k]=t
	#Method 1
	x=range(num_method)
	w=0.5
	fig, ax = plt.subplots(2,3,figsize=(30,10))
	for nn in range(len(N)):
		t_now=0
		i=nn/3
		j=nn%3
		ax[i,j].bar(x[0],t_methods[nn,0,0],width=w,color='darkblue',label='1,2,3-Spread')
		t_now+=t_methods[nn,0,0]
		ax[i,j].bar(x[0],t_methods[nn,0,1],w,bottom=t_now,color='crimson',label='1,2,3-Other')
	
		#Method2
		t_now=0
		ax[i,j].bar(x[1],t_methods[nn,1,4],w,bottom=t_now,color='darkblue')
		t_now+=t_methods[nn,1,4]
		ax[i,j].bar(x[1],t_methods[nn,1,1],w,bottom=t_now,color='slateblue',label='2-Create SortIdx; 3-Calculate Binsize')
		t_now+=t_methods[nn,1,1]
		ax[i,j].bar(x[1],t_methods[nn,1,2],w,t_now,color='blueviolet',label='2-Sort; 3-Scan Binsizearray')
		t_now+=t_methods[nn,1,2]
		ax[i,j].bar(x[1],t_methods[nn,1,3],w,t_now,color='violet',label='2,3-Pts Rearrange; 4,5-CalcInvofGlobalSortIdx')
		t_now+=t_methods[nn,1,3]
		ax[i,j].bar(x[1],t_methods[nn,1,0],w,t_now,color='fuchsia',label='2,3-CUDA malloc')
		t_now+=t_methods[nn,1,0]
		ax[i,j].bar(x[1],t_methods[nn,1,5],w,t_now,color='deeppink',label='2,3-CUDA Free')
		t_now+=t_methods[nn,1,5]
		ax[i,j].bar(x[1],t_methods[nn,1,6],w,t_now,color='crimson')
		
		#Method3
		t_now=0
		ax[i,j].bar(x[2],t_methods[nn,2,4],w,bottom=t_now,color='darkblue')
		t_now+=t_methods[nn,2,4]
		ax[i,j].bar(x[2],t_methods[nn,2,1],w,bottom=t_now,color='slateblue')
		t_now+=t_methods[nn,2,1]
		ax[i,j].bar(x[2],t_methods[nn,2,2],w,t_now,color='blueviolet')
		t_now+=t_methods[nn,2,2]
		ax[i,j].bar(x[2],t_methods[nn,2,3],w,t_now,color='violet')
		t_now+=t_methods[nn,2,3]
		ax[i,j].bar(x[2],t_methods[nn,2,0],w,t_now,color='fuchsia')
		t_now+=t_methods[nn,2,0]
		ax[i,j].bar(x[2],t_methods[nn,2,5],w,t_now,color='deeppink')
		t_now+=t_methods[nn,2,5]
		ax[i,j].bar(x[2],t_methods[nn,2,6],w,t_now,color='crimson')

		#Method4
		t_now=0
		ax[i,j].bar(x[3],t_methods[nn,3,5],w,bottom=t_now,color='darkblue')
		t_now+=t_methods[nn,3,5]
		ax[i,j].bar(x[3],t_methods[nn,3,1],w,bottom=t_now,color='slateblue')
		t_now+=t_methods[nn,3,1]
		ax[i,j].bar(x[3],t_methods[nn,3,2],w,t_now,color='blueviolet')
		t_now+=t_methods[nn,3,2]
		ax[i,j].bar(x[3],t_methods[nn,3,3],w,t_now,color='violet')
		t_now+=t_methods[nn,3,3]
		ax[i,j].bar(x[3],t_methods[nn,3,4],w,t_now,color='hotpink',label='Subproblem to Bin map')
		t_now+=t_methods[nn,3,4]
		ax[i,j].bar(x[3],t_methods[nn,3,0],w,t_now,color='fuchsia')
		t_now+=t_methods[nn,3,0]
		ax[i,j].bar(x[3],t_methods[nn,3,6],w,t_now,color='deeppink')
		t_now+=t_methods[nn,3,6]
		ax[i,j].bar(x[3],t_methods[nn,3,7],w,t_now,color='crimson')
		
		t_now=0
		ax[i,j].bar(x[4],t_methods[nn,4,5],w,bottom=t_now,color='darkblue')
		t_now+=t_methods[nn,4,5]
		ax[i,j].bar(x[4],t_methods[nn,4,1],w,bottom=t_now,color='slateblue')
		t_now+=t_methods[nn,4,1]
		ax[i,j].bar(x[4],t_methods[nn,4,2],w,t_now,color='blueviolet')
		t_now+=t_methods[nn,4,2]
		ax[i,j].bar(x[4],t_methods[nn,4,3],w,t_now,color='violet')
		t_now+=t_methods[nn,4,3]
		ax[i,j].bar(x[4],t_methods[nn,4,4],w,t_now,color='hotpink')
		t_now+=t_methods[nn,4,4]
		ax[i,j].bar(x[4],t_methods[nn,4,0],w,t_now,color='fuchsia')
		t_now+=t_methods[nn,4,0]
		ax[i,j].bar(x[4],t_methods[nn,4,6],w,t_now,color='deeppink')
		t_now+=t_methods[nn,4,6]
		ax[i,j].bar(x[4],t_methods[nn,4,7],w,t_now,color='crimson')

		ax[i,j].set_xticks(np.array(x)+0.5*w)
		ax[i,j].set_xlim([x[0]-0.5*w,x[-1]+1.5*w])
    		ax[i,j].set_xticklabels(['I', 'I_sort', 'I_binsort', 'Sp', 'Sp-indirect'])
		ax[i,j].set_xlabel('N={:3d}'.format(N[nn]))
		ax[i,j].set_ylabel('ms')
		ax[i,j].grid()

	fig.subplots_adjust(top=0.85)
	handles, labels = ax[-1,-2].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', ncol=4)

	plt.savefig('../timebreakdown_gpu_indirect_0802.pdf')
	plt.show()
if __name__== "__main__":
  main()
