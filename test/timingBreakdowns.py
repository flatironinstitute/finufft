#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import subprocess
import searchForTimeMetrics as stm

'''
A script to run a seris of finufftGuru_tests with varying parameters
Captures the stdout, parses it for timing statistics, and graphs speedup
ratio trends. 
'''


M_srcpts = 1e7 #1e6
tolerance = 1e-6
debug = 1

#modes array.
#first dimension counts are in modes[0,1,2]
#second dimension: modes[3,4,5]
#3rd dimension [6,7,8]
modes = [1e6,1,1,1e3,1e3,1,1e2,1e2,1e2]

dimensions = [1,2,3]
types = [1,2,3] 
#n_trials=[1,10,100]
n_trials = [1,10]


#data capture arrays
#Datapoints are speedup ratio := oldImplementationTime/newImplentationTime
totalTimeT1Ratio=[]
totalTimeT2Ratio=[]
totalTimeT3Ratio=[]
spreadT1Ratio=[]
spreadT2Ratio=[]
spreadT3Ratio=[]
fftwPlanT1Ratio=[]
fftwPlanT2Ratio=[]
fftwPlanT3Ratio=[]
fftT1Ratio=[]
fftT2Ratio=[]
fftT3Ratio=[]

#raw timing values

totalTimeT1_Old =[]
totalTimeT1_New =[]
totalTimeT2_Old =[]
totalTimeT2_New =[]
totalTimeT3_Old =[]
totalTimeT3_New =[]

spreadT1_Old =[]
spreadT1_New =[]
spreadT2_Old =[]
spreadT2_New =[]
spreadT3_Old =[]
spreadT3_New =[]

fftwPlanT1_Old =[]
fftwPlanT1_Old_initial =[]
fftwPlanT1_New =[]
fftwPlanT2_Old =[]
fftwPlanT2_Old_initial =[]
fftwPlanT2_New =[]
fftwPlanT3_Old =[]
fftwPlanT3_Old_initial =[]
fftwPlanT3_New =[]


fftT1_Old =[]
fftT1_New =[]
fftT2_Old =[]
fftT2_New =[]
fftT3_Old =[]
fftT3_New =[]

#do
for dim in dimensions:
    for ftype in types:
        for trial in n_trials:
            
            print( "./finufftGuru_test "+ str(trial)+ " " + str(ftype)+  " " +str(dim)+ " " + 
                                  str(modes[3*(dim-1)])+ " " + str(modes[3*(dim-1)+1])+ " " +  str(modes[3*(dim-1)+2])+ " " + 
                                  str(M_srcpts)+  " " + str(tolerance) + " " + str(debug)); 
            #execute the test for this set of parameters
            out =  subprocess.run(["./finufftGuru_test", str(trial), str(ftype), str(dim), 
                                  str(modes[3*(dim-1)]),str(modes[3*(dim-1)+1]),  str(modes[3*(dim-1)+2]), 
                                  str(M_srcpts), str(tolerance), str(debug)], 
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            strOut = out.stdout.decode() #convert bytes to string
            print(strOut)

            
            #parse the output and syphon into data arrays

            ###############################################################################

            #gather total plan,setpts,execute,destroy
            planTime = stm.extractTime('(finufft_plan.*completed)(.*)',strOut)
            setPtsTime = stm.extractTime('(finufft_setpts.*completed)(.*)',strOut)
            execTime = stm.extractTime('(finufft_exec.*completed)(.*)', strOut)
            delTime = stm.extractTime('(finufft_destroy.*completed)(.*)',strOut)
            totalNewTime = round(planTime + setPtsTime + execTime + delTime,5)

            #gather old total time
            totalOldTime = stm.extractTime('(execute.*in)(.*)(or .*)' ,strOut)

            if(ftype == 1):
                totalTimeT1_New.append(totalNewTime)
                totalTimeT1_Old.append(totalOldTime)
            elif(ftype == 2):
                totalTimeT2_New.append(totalNewTime)
                totalTimeT2_Old.append(totalOldTime)
            else:
                totalTimeT3_New.append(totalNewTime)
                totalTimeT3_Old.append(totalOldTime)

            #total time speedup
            totalSpeedup = round(totalOldTime/totalNewTime,5)

            if(ftype == 1):
                totalTimeT1Ratio.append(totalSpeedup)
            elif(ftype == 2):
                totalTimeT2Ratio.append(totalSpeedup)
            else:
                totalTimeT3Ratio.append(totalSpeedup)

            ###############################################################################
                
            #spread (old) / [sort+spread]  (new)            
            newSort = stm.sumAllTime('(.*finufft_setpts.*sort)(.*)', strOut)

            #collect spreading if any
            newSpread = stm.extractTime('(.*finufft.*exec.*spread)(.*)' , strOut) 

            #collect interp if any
            newInterp = stm.extractTime('(.*finufft.*exec.*interp)(.*)',strOut)

            #collect the spread timings for each trial of old
            totalOldSpread = stm.sumAllTime('(.*spread.*ier)(.*)', strOut) #gets spread AND unspread (i.e. interpolation)

            spreadRatio = round(totalOldSpread/(newSort + newSpread + newInterp),5)

            if(ftype == 1):
                spreadT1Ratio.append(spreadRatio)
                spreadT1_New.append(round(newSort + newSpread + newInterp,5))
                spreadT1_Old.append(totalOldSpread)
            elif(ftype == 2):
                spreadT2Ratio.append(spreadRatio)
                spreadT2_New.append(round(newSort + newSpread + newInterp,5))
                spreadT2_Old.append(totalOldSpread)
            else:
                spreadT3Ratio.append(spreadRatio)
                spreadT3_New.append(round(newSort + newSpread + newInterp,5))
                spreadT3_Old.append(totalOldSpread)

            ###############################################################################

            #fftw_plan(old) / fftw_plan(new)
            planSciNotString = '(\(\d+\)[ \t]+)(\d*.?\d*e-\d* s)'
            planDecimalMatchString= '(\(\d+\)[ \t]+)(\d*\.?\d* s)'
            planWholeNumberMatchString= '(\(\d+\)[ \t]+)(\d+ s)' 

            #collect new fftw_plan time
            new_fftwPlan=0
            lineMatch = re.search("(.*make plan.*fftw plan \(\d+\).*)",strOut)
            if(lineMatch):
                fftwPlanVal = re.search(planSciNotString, lineMatch.group(0))
                if(not fftwPlanVal):
                    fftwPlanVal = re.search(planDecimalMatchString, lineMatch.group(0))
                if(not fftwPlanVal):
                    fftwPlanVal = re.search(wholeNumberMatchString, lineMatch.group(0))
                new_fftwPlan = float(fftwPlanVal.group(2).split('s')[0])  #strip off s
            new_fftwPlan = round(new_fftwPlan,5)    

            #collect the fftw_plan timings for each trial of old
            isInitial = True
            initialLookup = 0
            totalOldfftwPlan=0   
            lineMatch = re.findall('(?<!\[make plan\] )fftw plan \(64\).*', strOut) #all fftw plan lines that don't include "make plan" indicating old implm.
            if(lineMatch):
                for match in lineMatch:
                    oldfftwPlanVal = re.search(planSciNotString, match)
                    if(not oldfftwPlanVal):
                        oldfftwPlanVal = re.search(planDecimalMatchString, match)
                    if(not oldfftwPlanVal):
                        oldfftwPlanVal = re.search(planWholeNumberMatchString, match)
                    oldfftwPlanVal = float(oldfftwPlanVal.group(2).split('s')[0]) #trim off " s"
                    if(isInitial): #Capture the first fftwplan output - indicating initial construction time
                        initalLookup = oldfftwPlanVal
                        isInitial = False
                    
                    totalOldfftwPlan = totalOldfftwPlan + oldfftwPlanVal
            totalOldfftwPlan = round(totalOldfftwPlan,5)
            
            #These plan ratios include the initial old implementation plan construction!!
            fftwPlanRatio = round(totalOldfftwPlan/new_fftwPlan,5)
            
            if(ftype == 1):
                fftwPlanT1Ratio.append(fftwPlanRatio)
                fftwPlanT1_New.append(new_fftwPlan)
                fftwPlanT1_Old.append(totalOldfftwPlan)
                fftwPlanT1_Old_initial.append(initialLookup)
            elif(ftype == 2):
                fftwPlanT2Ratio.append(fftwPlanRatio)
                fftwPlanT2_New.append(new_fftwPlan)
                fftwPlanT2_Old.append(totalOldfftwPlan)
                fftwPlanT2_Old_initial.append(initialLookup)
            else:
                fftwPlanT3Ratio.append(fftwPlanRatio)
                fftwPlanT3_New.append(new_fftwPlan)
                fftwPlanT3_Old.append(totalOldfftwPlan)
                fftwPlanT3_Old_initial.append(initialLookup)
            
            ###############################################################################
            #fftw_exec(old) / fftw_exec(new)

            #collect new fft time
            new_fft = stm.extractTime("(.*finufft_exec.*fft)(.*)" , strOut)

            #collect the fftw_exec timings for each trial of old
            totalOldfft = stm.sumAllTime("(.*fft \(\d+ threads\))(.*)",strOut) 

            fftRatio = round(totalOldfft/new_fft,5)
            
            if(ftype == 1):
                fftT1Ratio.append(fftRatio)
                fftT1_New.append(new_fft)
                fftT1_Old.append(totalOldfft)
            elif(ftype == 2):
                fftT2Ratio.append(fftRatio)
                fftT2_New.append(new_fft)
                fftT2_Old.append(totalOldfft)
            else:
                fftT3Ratio.append(fftRatio)
                fftT3_New.append(new_fft)
                fftT3_Old.append(totalOldfft)

            
            ###############################################################################


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

print("n_trials:" )
print(t1x)
print("dimensions: ")
print(t1y)
    
zbot = np.zeros(len(t1x))
widths = [barWidth]*len(t1x)
depths = [barDepth]*len(t1x)

fig = plt.figure()
#create legend
t1_proxy = plt.Rectangle((0,0),1,1,fc="r")
t2_proxy = plt.Rectangle((0,0),1,1,fc="b")
t3_proxy = plt.Rectangle((0,0),1,1,fc="g")


##################TotalSpeed BAR GRAPH####################################################
print("##################Total Time####################################################") 
print("\n")
print("Raw T1 Total Time New " + str(totalTimeT1_New))
print("Raw T1 Total Time Old " + str(totalTimeT1_Old))
print("TotalTime T1Ratio " + str(totalTimeT1Ratio))
print("\n")
print("Raw T2 Total Time New " + str(totalTimeT2_New))
print("Raw T2 Total Time Old " + str(totalTimeT2_Old))
print("TotalTime T2Ratio " + str(totalTimeT2Ratio))
print("\n")
print("Raw T3 Total Time New " + str(totalTimeT3_New))
print("Raw T3 Total Time Old " + str(totalTimeT3_Old))
print("TotalTime T3Ratio " + str(totalTimeT3Ratio))
print("\n")
ax1 = fig.add_subplot(221,projection='3d')

if(totalTimeT1Ratio):
    ax1.bar3d(t1x, t1y, zbot, widths, depths, totalTimeT1Ratio, shade=True, color='r', label='type1', alpha='1')
if(totalTimeT2Ratio):
    ax1.bar3d(t2x, t2y, zbot, widths, depths, totalTimeT2Ratio, shade=True, color='b', label='type2', alpha='1')
if(totalTimeT3Ratio):
    ax1.bar3d(t3x, t3y, zbot, widths, depths, totalTimeT3Ratio, shade=True, color='g', label='type3', alpha='1')

ax1.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('totalOldTime/totalNewTime')


#### Speed Statistics SANS Initial Planning Time 

TotalSpeedRatioSansPlanT1 = (np.array(totalTimeT1_Old) - np.array(fftwPlanT1_Old_initial))/(np.array(totalTimeT1_New) - np.array(fftwPlanT1_New))
TotalSpeedRatioSansPlanT2 = (np.array(totalTimeT2_Old) - np.array(fftwPlanT2_Old_initial))/(np.array(totalTimeT2_New) - np.array(fftwPlanT2_New))
TotalSpeedRatioSansPlanT3 = (np.array(totalTimeT3_Old) - np.array(fftwPlanT3_Old_initial))/(np.array(totalTimeT3_New) - np.array(fftwPlanT3_New))


print("Total Speed Ratio SANS initial fftwPlan for old implm. and only fftwPlan for new")
print("T1: " + str(TotalSpeedRatioSansPlanT1))
print("T2: " + str(TotalSpeedRatioSansPlanT2))
print("T3: " + str(TotalSpeedRatioSansPlanT3))






##################SPREADING BAR GRAPH####################################################
print("##################SPREADING####################################################") 
print("\n")
print("Raw T1 Spreading New" + str(spreadT1_New))
print("Raw T1 Spreading Old" + str(spreadT1_Old))
print("Spreading T1Ratio " + str(spreadT1Ratio))
print("\n")
print("Raw T2 Spreading New" + str(spreadT2_New))
print("Raw T2 Spreading Old" + str(spreadT2_Old))
print("Spreading T2Ratio " + str(spreadT2Ratio))
print("\n")
print("Raw T3 Spreading New" + str(spreadT3_New))
print("Raw T3 Spreading Old" + str(spreadT3_Old))
print("Spreading T3Ratio " + str(spreadT3Ratio))
print("\n")

ax2 = fig.add_subplot(222,projection='3d')

if(spreadT1Ratio):
    ax2.bar3d(t1x, t1y, zbot, widths, depths, spreadT1Ratio, shade=True, color='r', label='type1', alpha='1')
if(spreadT2Ratio):
    ax2.bar3d(t2x, t2y, zbot, widths, depths, spreadT2Ratio, shade=True, color='b', label='type2', alpha='1')
if(spreadT3Ratio):
    ax2.bar3d(t3x, t3y, zbot, widths, depths, spreadT3Ratio, shade=True, color='g', label='type3', alpha='1')

ax2.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('oldSpreadTime/NewSpreadTime')



##################Total FFTW Plan BAR GRAPH####################################################
print("##################Total FFTW Plan####################################################")
print("\n")
print("FFTW Plan T1 New " + str(fftwPlanT1_New))
print("FFTW Plan T1 Old " + str(fftwPlanT1_Old))
print("FFTW Plan T1Ratio " + str(fftwPlanT1Ratio))
print("\n")
print("FFTW Plan T2 New " + str(fftwPlanT2_New))
print("FFTW Plan T2 Old " + str(fftwPlanT2_Old))
print("FFTW Plan T2Ratio " + str(fftwPlanT2Ratio))
print("\n")
print("FFTW Plan T3 New " + str(fftwPlanT3_New))
print("FFTW Plan T3 Old " + str(fftwPlanT3_Old))
print("FFTW Plan T3Ratio " + str(fftwPlanT3Ratio))
print("\n")

ax3 = fig.add_subplot(223,projection='3d')

if(fftwPlanT1Ratio):
    ax3.bar3d(t1x, t1y, zbot, widths, depths, fftwPlanT1Ratio, shade=True, color='r', label='type1', alpha='1')
if(fftwPlanT2Ratio):
    ax3.bar3d(t2x, t2y, zbot, widths, depths, fftwPlanT2Ratio, shade=True, color='b', label='type2', alpha='1')
if(fftwPlanT3Ratio):
    ax3.bar3d(t3x, t3y, zbot, widths, depths, fftwPlanT3Ratio, shade=True, color='g', label='type3', alpha='1')

ax3.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('totalOldFFtwPlan/NewFftwPlan')


##################Total FFT Exec BAR GRAPH####################################################
print("##################Total FFT Exec####################################################")
print("\n")
print("FFT Exec T1 New " + str(fftT1_New))
print("FFT Exec T1 Old " + str(fftT1_Old))
print("FFT Exec T1Ratio " + str(fftT1Ratio))
print("\n")
print("FFT Exec T2 New " + str(fftT2_New))
print("FFT Exec T2 Old " + str(fftT2_Old))
print("FFT Exec T2Ratio " + str(fftT2Ratio))
print("\n")
print("FFT Exec T3 New " + str(fftT3_New))
print("FFT Exec T3 Old " + str(fftT3_Old))
print("FFT Exec T3Ratio " + str(fftT3Ratio))
print("\n")

ax4 = fig.add_subplot(224,projection='3d')

if(fftT1Ratio):
    ax4.bar3d(t1x, t1y, zbot, widths, depths, fftT1Ratio, shade=True, color='r', label='type1', alpha='1')
if(fftT2Ratio):
    ax4.bar3d(t2x, t2y, zbot, widths, depths, fftT2Ratio, shade=True, color='b', label='type2', alpha='1')
if(fftT3Ratio):
    ax4.bar3d(t3x, t3y, zbot, widths, depths, fftT3Ratio, shade=True, color='g', label='type3', alpha='1')

ax4.legend([t1_proxy,t2_proxy,t3_proxy], ['type1','type2','type3'])

plt.xlabel('n_trials')
plt.ylabel('Dimensions')
plt.yticks([y+barWidth+1 for y in range(len(t1y))], ['1', '2', '3'])
plt.title('totalOldFFtwExec/NewFftwExec')

plt.show()

fig.savefig('timing_breakdowns_rusty_node.png')












