#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import re
import subprocess

M_srcpts = 1e5
tolerance = 1e-6
debug = 1
modes = [1e5,1,1,1e2,1e2,1,1e2,1e2,1e1]
dimensions = [1,2,3]
types = [1,2,3] 
n_trials = [1,10,100]


#data capture arrays
#Ratio = oldImplementation/newImplentation

#import the data

(totalTimeT1, totalTimeT2,totalTimeT3, spreadT1, spreadT2, spreadT3, fftwPlanT1, fftwPlanT2, fftwPlanT3, fftT1, fftT2, fftT3)  = np.loadtxt('finale.data', unpack=True, dtype=[('totalT1','float'), ('totalT2','float'), ('totalT3','float'),
                                         ('spreadT1','float'), ('spreadT2','float'), ('spreadT3','float'),
                                         ('fftwPlanT1','float'), ('fftwPlanT2','float'), ('fftwPlanT3','float'),
                                         ('fftT1','float'), ('fftT2','float'), ('fftT3','float')])



##################Total FFTW Plan BAR GRAPH####################################################

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.bar(1, fftwPlanT1, label='type1')
ax1.bar(2, fftwPlanT2, label='type2')
ax1.bar(3, fftwPlanT3, label='type3')


plt.xlabel('Type')
plt.xticks([1,2,3],[1,2,3])
plt.title('1000 Trial totalOldFFtwPlan/NewFftwPlan Speedup')
plt.show()
fig.savefig('1000_Trial_totalOldFFtwPlanNewFftwPlan_Speedup.png')

##################Total FFT Exec BAR GRAPH####################################################
fig = plt.figure()

ax2 = fig.add_subplot(111)

ax2.bar(1, fftT1, label='type1')
ax2.bar(2, fftT2, label='type2')
ax2.bar(3, fftT3, label='type3')

plt.xlabel('Type')
plt.xticks([1,2,3],[1,2,3])

plt.title('1000_Trial_totalOldFFtwExec/NewFftwExec')
plt.show()
fig.savefig('1000_Trial_totalfftwExecNewfftwExec.png')



##################SPREADING BAR GRAPH####################################################

fig = plt.figure()

ax3 = fig.add_subplot(111)

ax3.bar(1, spreadT1, label='type1')
ax3.bar(2, spreadT2, label='type2')
ax3.bar(3, spreadT3, label='type3')

plt.xlabel('Type')
plt.xticks([1,2,3],[1,2,3])


plt.title('1000_Trial_oldSpreadTime/NewSpreadTime')
plt.show()
fig.savefig('1000_Trial_oldSpreadTime-NewSpreadTime.png')


##################TotalSpeed BAR GRAPH####################################################


fig = plt.figure()

ax4 = fig.add_subplot(111)

ax4.bar(1, totalTimeT1, label='type1')
ax4.bar(2, totalTimeT2, label='type2')
ax4.bar(3, totalTimeT3, label='type3')

plt.xlabel('Type')
plt.xticks([1,2,3],[1,2,3])


plt.title('1000_Trial_totalOldTime/totalNewTime')
plt.show()
fig.savefig('1000_Trial_TotalOldTime-TotalNewTime.png')




