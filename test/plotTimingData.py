#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

(totalTimeT1, totalTimeT2,totalTimeT3, spreadT1, spreadT2, spreadT3, fftwPlanT1, fftwPlanT2, fftwPlanT3, fftT1, fftT2, fftT3)  = np.loadtxt('timing.data', unpack=True, dtype=[('totalT1','float'), ('totalT2','float'), ('totalT3','float'),
                                         ('spreadT1','float'), ('spreadT2','float'), ('spreadT3','float'),
                                         ('fftwPlanT1','float'), ('fftwPlanT2','float'), ('fftwPlanT3','float'),
                                         ('fftT1','float'), ('fftT2','float'), ('fftT3','float')])



#Construct the bar graph 
barWidth = 0.25
barDepth = 0.25

_t1ys = dimensions
_t2ys = [y + barWidth for y in _t1ys]
_t3ys = [y + barWidth for y in _t2ys]

_t1xs = n_trials 
_t2xs = [x + barDepth for x in _t1xs]
_t3xs = [x + barDepth for x in _t2xs]

_t1xx, _t1yy = np.meshgrid(_t1xs,_t1ys)
t1x,t1y = _t1xx.ravel(), _t1yy.ravel()

_t2xx, _t2yy = np.meshgrid(_t2xs,_t2ys)
t2x,t2y = _t2xx.ravel(), _t2yy.ravel()

_t3xx, _t3yy = np.meshgrid(_t3xs,_t3ys)
t3x,t3y = _t3xx.ravel(), _t3yy.ravel()

print(t1x)
print(t1y)
    
zbot = np.zeros(len(t1x))
widths = [barWidth]*len(t1x)
depths = [barDepth]*len(t1x)

fig = plt.figure()
#create legend
t1_proxy = plt.Rectangle((0,0),1,1,fc="r")
t2_proxy = plt.Rectangle((0,0),1,1,fc="b")
t3_proxy = plt.Rectangle((0,0),1,1,fc="g")


##################Total FFTW Plan BAR GRAPH####################################################

print("FFTW Plan T1 " + str(fftwPlanT1))
print("FFTW Plan T2 " + str(fftwPlanT2))
print("FFTW Plan T3 " + str(fftwPlanT3))

fig = plt.figure()

logfftwPlanT1 = np.zeros(len(fftwPlanT1))
logfftwPlanT2 = np.zeros(len(fftwPlanT1))
logfftwPlanT3 = np.zeros(len(fftwPlanT1))
for i in range(len(fftwPlanT1)):
    logfftwPlanT1 = math.log(fftwPlanT1[i])
    logfftwPlanT2 = math.log(fftwPlanT2[i])
    logfftwPlanT3 = math.log(fftwPlanT3[i])

ax3 = fig.add_subplot(111,projection='3d')

ax3.bar3d(t1x, t1y, zbot, widths, depths, fftwPlanT1, shade=False, color='r', label='type1', alpha='1')
ax3.bar3d(t2x, t2y, zbot, widths, depths, fftwPlanT2, shade=False, color='b', label='type2', alpha='1')
ax3.bar3d(t3x, t3y, zbot, widths, depths, fftwPlanT3, shade=False, color='g', label='type3', alpha='1')

ax3.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('(totalOldFFtwPlan/NewFftwPlan)')
plt.show()
fig.savefig('totalOldfftwPlan-NewfftwPlan.png')

##################Total FFT Exec BAR GRAPH####################################################

print("FFT Exec T1 " + str(fftT1))
print("FFT Exec T2 " + str(fftT2))
print("FFT Exec T3 " + str(fftT3))

fig = plt.figure()

logfftT1 = np.zeros(len(fftT1))
logfftT2 = np.zeros(len(fftT1))
logfftT3 = np.zeros(len(fftT1))
for i in range(len(fftT1)):
    logfftT1 = math.log(fftT1[i])
    logfftT2 = math.log(fftT2[i])
    logfftT3 = math.log(fftT3[i])


ax4 = fig.add_subplot(111,projection='3d')

ax4.bar3d(t1x, t1y, zbot, widths, depths, fftT1, shade=True, color='r', label='type1', alpha='1')
ax4.bar3d(t2x, t2y, zbot, widths, depths, fftT2, shade=True, color='b', label='type2', alpha='1')
ax4.bar3d(t3x, t3y, zbot, widths, depths, fftT3, shade=True, color='g', label='type3', alpha='1')

ax4.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('totalOldFFtwExec/NewFftwExec')
plt.show()
fig.savefig('totalfftwExec/newfftwExec')



##################SPREADING BAR GRAPH####################################################
print("Spreading T1 " + str(spreadT1))
print("Spreading T2 " + str(spreadT2))
print("Spreading T3 " + str(spreadT3))

fig = plt.figure()

logSpreadT1 = np.zeros(len(spreadT1))
logSpreadT2 = np.zeros(len(spreadT1))
logSpreadT3 = np.zeros(len(spreadT1))
for i in range(len(spreadT1)):
    logSpreadT1 = math.log(spreadT1[i])
    logSpreadT2 = math.log(spreadT2[i])
    logSpreadT3 = math.log(spreadT3[i])


ax2 = fig.add_subplot(111,projection='3d')

ax2.bar3d(t1x, t1y, zbot, widths, depths, spreadT1, shade=False, color='r', label='type1', alpha='1')
ax2.bar3d(t2x, t2y, zbot, widths, depths, spreadT2, shade=False, color='b', label='type2', alpha='1')
ax2.bar3d(t3x, t3y, zbot, widths, depths, spreadT3, shade=False, color='g', label='type3', alpha='1')

ax2.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('oldSpreadTime/NewSpreadTime')
plt.show()
fig.savefig('oldSpreadTime-NewSpreadTime.png')


##################TotalSpeed BAR GRAPH####################################################
print("TotalTime T1 " + str(totalTimeT1))
print("TotalTime T2 " + str(totalTimeT2))
print("TotalTime T3 " + str(totalTimeT3))

ax1 = fig.add_subplot(111,projection='3d')

logTotalTimeT1 = np.zeros(len(totalTimeT1))
logTotalTimeT2 = np.zeros(len(totalTimeT1))
logTotalTimeT3 = np.zeros(len(totalTimeT1))
for i in range(len(totalTimeT1)):
    logTotalTimeT1 = math.log(totalTimeT1[i])
    logTotalTimeT2 = math.log(totalTimeT2[i])
    logTotalTimeT3 = math.log(totalTimeT3[i])
    

ax1.bar3d(t1x, t1y, zbot, widths, depths, totalTimeT1, shade=False, color='r', label='type1', alpha='1')
ax1.bar3d(t2x, t2y, zbot, widths, depths, totalTimeT2, shade=False, color='b', label='type2', alpha='1')
ax1.bar3d(t3x, t3y, zbot, widths, depths, totalTimeT3, shade=False, color='g', label='type3', alpha='1')

ax1.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('totalOldTime/totalNewTime')
plt.show()
fig.savefig('TotalOldTime-TotalNewTime.png')











