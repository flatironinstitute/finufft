import subprocess
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def main():
	filename='../results/spreadwidth/'
	num_lines = sum(1 for line in open(filename+'cufinufft_tol_1e-6.out', 'r'))-1
	N = np.zeros(num_lines, dtype=int)
	t_cunfft = np.zeros(num_lines)
	t_nfft = np.zeros(num_lines)
	t_cufinufft = np.zeros([1, num_lines])
	t_finufft = np.zeros(num_lines)
	#speedup = np.zeros([4, num_lines])
	f = open(filename+'cunfft_tol_1e-6.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 'N=', ',')
		if(temp==""):
                        continue
		else:
        		N[i]= int(temp)
			t_cunfft[i]=float(find_between(line, 't=', '\n'))
			i=i+1

	f = open(filename+'cufinufft_tol_1e-6.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_cufinufft[i/num_lines, i%num_lines]=float(temp)
			i=i+1

	f = open(filename+'finufft_tol_1e-6.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_finufft[i]=float(temp)
			i=i+1

	f = open(filename+'nfft_tol_1e-6.out', 'r')
	i=0
	for line in f:
		temp=find_between(line, 't=', '\n')
		if(temp==""):
			continue
		else:
			t_nfft[i]=float(temp)
			i=i+1
	"""
	#for m in range(4):
	#	speedup[m,:] = np.divide(t_cunfft, t_finufft[m,:])
	x = 0.0
	w = 0.02
	fig, ax= plt.subplots(1,1,figsize=(10, 5))
	nn=-1
	M = (N[-1]/2.0)**2*1000
	ax2 = ax.twinx()
	xx = [x+0.5*w,x+1.5*w,x+2.5*w,x+3.5*w]
	yy = [t_finufft[nn]/t_nfft[nn], 1.0, t_finufft[nn]/t_cunfft[nn], t_finufft[nn]/t_cufinufft[0,nn]]
	ax2.plot(xx,yy,'-kx')
	ax2.set_ylabel('speedup')
	ax2.axhline(y=1, linestyle='--', color='r',label='finufft')

	ax.bar(x, M/t_nfft[nn], w, color='darkblue',log=0, label='nfft')
	ax.bar(x+w, M/t_finufft[nn], w, color='slateblue',log=0, label='finufft')
	ax.bar(x+2*w, M/t_cunfft[nn], w, color='blueviolet',log=0, label='cunfft')
	ax.bar(x+3*w, M/t_cufinufft[0,nn], w,color='darkviolet',log=0, label='GPU: Subprob')
	#ax[i,j].bar(x+4*w, t_cufinufft[1,nn], w, color='violet',log=1, label='GPU: input driven with sort')
	#ax[i,j].bar(x+5*w, t_cufinufft[2,nn], w, color='deeppink',log=1, label='GPU: hybrid')
	#ax[i,j].bar(x+6*w, t_cufinufft[3,nn], w, color='lightpink',log=1, label='GPU: subprob')

	ax.set_xticks([x+2*w])
	ax.set_xticklabels(N[nn:nn+1])
	ax.set_xlim([x-0.5*w, x+4.5*w])
	ax.set_ylabel('#NU pts/s')
	ax.set_xlabel('N')
	ax.grid()
	#ax[i,j].set_ylim(ymin=0.1)
	from matplotlib import ticker
	formatter = ticker.ScalarFormatter()
                #ax[i,j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
                #ax[i,j].set_ylim((0, 12))
        plt.tight_layout(rect=[0.01, 0.07, 1, 1], pad=1.5, w_pad=5.0, h_pad=1.5)
        fig.suptitle('Clustered nupts, Double Precision, w = 7')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=6)
        plt.savefig('../speed_all_N_4096_clusterednupts_tol_1e-6.pdf')
        plt.show()
	"""
	"""
	fig, ax = plt.subplots(figsize=(30,10))
	x = range(num_lines)
	w = 1
	ax.semilogy(x, t_nfft, '-o', label='nfft')
	ax.semilogy(x, t_finufft, '-o', label='finufft')
	ax.semilogy(x, t_cunfft, '-o', label='cunfft')
	ax.semilogy(x, t_cufinufft[0,:], '-o', label='cufinufft')
	ax.set_xticks(x)
	ax.set_xticklabels(N)
	ax.set_xlim([x[0]-0.5*w, x[-1]+0.5*w])
	ax.set_ylabel('ms')
	ax.set_xlabel('N')
	ax.set_title('Spread time for clustered nupts')
	ax.grid()
        plt.legend(loc=0)
	"""
	
	def format_e(n):
    		a = '%E' % n
    		return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

	x = 0.0
	w = 0.02
	fig, ax= plt.subplots(2,3,figsize=(30, 10))
	for nn in range(len(N)):
		i=nn/3
		j=nn%3
		M = (N[nn]/2.0)**2*1000
		ax2 = ax[i,j].twinx()
		xx = [x+0.5*w,x+1.5*w,x+2.5*w,x+3.5*w]
		yy = [t_finufft[nn]/t_nfft[nn], 1.0, t_finufft[nn]/t_cunfft[nn], t_finufft[nn]/t_cufinufft[0,nn]]
		ax2.plot(xx,yy,'-ko')
		ax2.axhline(y=1, linestyle='--', color='r',label='finufft')

		ax[i,j].bar(x, M/t_nfft[nn], w, color='darkblue',log=0, label='nfft')
		ax[i,j].bar(x+w, M/t_finufft[nn], w, color='slateblue',log=0, label='finufft')
		ax[i,j].bar(x+2*w, M/t_cunfft[nn], w, color='blueviolet',log=0, label='cunfft')
		ax[i,j].bar(x+3*w, M/t_cufinufft[0,nn], w,color='darkviolet',log=0, label='GPU: Subprob')

		h = 1e6
		ax[i,j].text(x+0.4*w,M/t_nfft[nn]+h, "{:.2e}".format(M/t_nfft[nn]))
		ax[i,j].text(x+1.4*w,M/t_finufft[nn]+h, "{:.2e}".format(M/t_finufft[nn]))
		ax[i,j].text(x+2.4*w,M/t_cunfft[nn]+h, "{:.2e}".format(M/t_cunfft[nn]))
		ax[i,j].text(x+3.4*w,M/t_cufinufft[0,nn]+h, "{:.2e}".format(M/t_cufinufft[0,nn]))
		#ax[i,j].bar(x+4*w, t_cufinufft[1,nn], w, color='violet',log=1, label='GPU: input driven with sort')
		#ax[i,j].bar(x+5*w, t_cufinufft[2,nn], w, color='deeppink',log=1, label='GPU: hybrid')
		#ax[i,j].bar(x+6*w, t_cufinufft[3,nn], w, color='lightpink',log=1, label='GPU: subprob')

		ax[i,j].set_xticks([x+2*w])
		ax[i,j].set_xticklabels(N[nn:nn+1])
		ax[i,j].set_xlim([x-0.5*w, x+4.5*w])
		ax[i,j].set_ylabel('#NU pts/s')
		ax[i,j].set_xlabel('nf1')
		ax[i,j].grid()
		#ax[i,j].set_ylim(ymin=0.1)
		from matplotlib import ticker
		formatter = ticker.ScalarFormatter()
		formatter.set_scientific(True)
		formatter.set_powerlimits((-1,1))
		ax[i,j].yaxis.set_major_formatter(formatter)
		#ax[i,j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3g'))
		#ax[i,j].set_ylim((0, 12))
	plt.tight_layout(rect=[0.01, 0.07, 1, 1], pad=1.5, w_pad=5.0, h_pad=1.5)
	fig.suptitle('Clustered nupts, Double Precision, w = 7')	
	handles, labels = ax[0,0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=6)
	#plt.savefig('../speed_all_clusterednupts_tol_1e-6.pdf')
	plt.show()
	
if __name__== "__main__":
  main()
