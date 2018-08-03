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
	reps=1
	density = 1.0
	N_totry = 2**np.arange(7,13)
	t_gpuspread_1 = np.zeros([len(N_totry),2])
	t_gpuspread_2 = np.zeros([len(N_totry),7])
	t_gpuspread_3 = np.zeros([len(N_totry),7])
	t_gpuspread_4 = np.zeros([len(N_totry),7])
	for i,N in enumerate(N_totry):
		M = int((N/2.0)*(N/2.0))
		# Method 1
		t = np.zeros(2)
		for n in range(reps):
                        output=subprocess.check_output(["./compare",'1',str(nupts_distr),str(N),str(N)], \
                                            cwd="../../").decode("utf-8")
                        t[0]+= float(find_between(output, "Spread_2d_Idriven", "ms"))
                        t[1]+= float(find_between(output, "Spread\t\t", "ms"))
		for k in range(2):
			t_gpuspread_1[i,k] = t[k]/reps
		t_gpuspread_1[i,-1] -= sum(t_gpuspread_1[i,:-1])
		
		# Method 2
		t = np.zeros(7)
		bin_sort=0
		for n in range(reps):
                        output=subprocess.check_output(["./compare",'2',str(nupts_distr),str(N),str(N),str(M), \
                                                         '1e-6', '0', str(bin_sort)], cwd="../../").decode("utf-8")
                        t[0]+= float(find_between(output, "array", "ms"))
                        t[1]+= float(find_between(output, "CreateSortIdx", "ms"))
                        t[2]+= float(find_between(output, "CUB::SortPairs", "ms"))
                        t[3]+= float(find_between(output, "PtsRearrage", "ms"))
                        t[4]+= float(find_between(output, "Spread_2d_Idriven", "ms"))
                        t[5]+= float(find_between(output, "GPU-memory", "ms"))
                        t[6]+= float(find_between(output, "Spread\t\t", "ms"))
		for k in range(7):
			t_gpuspread_2[i,k] = t[k]/reps
		t_gpuspread_2[i,-1] -= sum(t_gpuspread_2[i,:-1])

		# Method 2 //bin_sort
		t = np.zeros(7)
		bin_sort=1
		for n in range(reps):
                        output=subprocess.check_output(["./compare",'2',str(nupts_distr),str(N),str(N),str(M), \
                                                         '1e-6', '0', str(bin_sort)], cwd="../../").decode("utf-8")
                        t[0]+= float(find_between(output, "array", "ms"))
                        t[1]+= float(find_between(output, "CalcBinSize_noghost_2d", "ms"))
                        t[2]+= float(find_between(output, "BinStartPts_2d", "ms"))
                        t[3]+= float(find_between(output, "PtsRearrange_noghost_2d", "ms"))
                        t[4]+= float(find_between(output, "Spread_2d_Idriven", "ms"))
                        t[5]+= float(find_between(output, "GPU-memory", "ms"))
                        t[6]+= float(find_between(output, "Spread\t\t", "ms"))
		for k in range(7):
			t_gpuspread_3[i,k] = t[k]/reps
		t_gpuspread_3[i,-1] -= sum(t_gpuspread_3[i,:-1])
		# Method 3
		#t = 0
		#for n in range(reps):
                #        output=subprocess.check_output(["./compare",'3',str(nupts_distr),str(N),str(N)], \
                #                            cwd="../../").decode("utf-8")
                #        t+= float(find_between(output, "Spread", "ms"))
		#t_gpuspread_3[i] = t/reps

		# Method 4
		t = np.zeros(7)
		for n in range(reps):
                        output=subprocess.check_output(["./compare",'4',str(nupts_distr),str(N),str(N)], \
                                            cwd="../../").decode("utf-8")
                        t[0]+= float(find_between(output, "array", "ms"))
                        t[1]+= float(find_between(output, "CalcBinSize_noghost_2d", "ms"))
                        t[2]+= float(find_between(output, "BinStartPts_2d", "ms"))
                        t[3]+= float(find_between(output, "PtsRearrange_noghost_2d", "ms"))
                        t[4]+= float(find_between(output, "Spread_2d_Hybrid", "ms"))
                        t[5]+= float(find_between(output, "GPU-memory", "ms"))
                        t[6]+= float(find_between(output, "Spread\t\t", "ms"))
		for k in range(7):
			t_gpuspread_4[i,k] = t[k]/reps
		t_gpuspread_4[i,-1] -= sum(t_gpuspread_4[i,:-1])
	
	# Output result
	print("Method 1: input driven without sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}'.format(N))
		for k in range(2):
			print('k={:d}\t{:5.3g}'.format(k, t_gpuspread_1[i,k]))
		#print('Spread           \t{:5.3g}'.format(t_gpuspread_1[i,0]))
		#print('Other            \t{:5.3g}'.format(t_gpuspread_1[i,1]))

	print("Method 2: input driven with sort")
	for i,N in enumerate(N_totry):
		print('N={:5d}'.format(N))
		for k in range(7):
			print('k={:d}\t{:5.3g}'.format(k, t_gpuspread_2[i,k]))
		#print('CUDA malloc      \t{:5.3g}'.format(t_gpuspread_2[i,0]))
		#print('Create SortIdx   \t{:5.3g}'.format(t_gpuspread_2[i,1]))
		#print('Sort             \t{:5.3g}'.format(t_gpuspread_2[i,2]))
		#print('Pts Rearrange    \t{:5.3g}'.format(t_gpuspread_2[i,3]))
		#print('Spread           \t{:5.3g}'.format(t_gpuspread_2[i,4]))
		#print('CUDA Free        \t{:5.3g}'.format(t_gpuspread_2[i,5]))
		#print('Other            \t{:5.3g}'.format(t_gpuspread_2[i,6]))

	print("Method 2: input driven with binsort")
	for i,N in enumerate(N_totry):
		print('N={:5d}'.format(N))
		for k in range(7):
			print('k={:d}\t{:5.3g}'.format(k, t_gpuspread_3[i,k]))
	#print("Method 3: output driven")
	#for i,N in enumerate(N_totry):
	#	print('N={:5d}'.format(N))
	#	for k in range(8):
	#		print('\t{:5.3g}'.format(t_gpuspread_3[i,k]))
	#print("\n")

	print("Method 4: hybrid")
	for i,N in enumerate(N_totry):
		print('N={:5d}'.format(N))
		for k in range(7):
			print('k={:d}\t{:5.3g}'.format(k, t_gpuspread_4[i,k]))
		#print('CUDA malloc      \t{:5.3g}'.format(t_gpuspread_4[i,0]))
		#print('Calculate Binsize\t{:5.3g}'.format(t_gpuspread_4[i,1]))
		#print('Scan binsizearray\t{:5.3g}'.format(t_gpuspread_4[i,2]))
		#print('Pts Rearrange    \t{:5.3g}'.format(t_gpuspread_4[i,3]))
		#print('Spread           \t{:5.3g}'.format(t_gpuspread_4[i,4]))
		#print('CUDA Free        \t{:5.3g}'.format(t_gpuspread_4[i,5]))
		#print('Other            \t{:5.3g}'.format(t_gpuspread_4[i,6]))
  
if __name__== "__main__":
  main()
