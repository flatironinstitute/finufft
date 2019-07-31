#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import subprocess

M_srcpts = 1e7
tolerance = 1e-6
debug = 1
modes = [1e6,1,1,1e3,1e3,1,1e2,1e2,1e2]
dimensions = [1,2,3]
types = [1,2,3]
n_trials = [1,10,100]


#data capture arrays
#Ratio = oldImplementation/newImplentation
totalTimeT1=[]
totalTimeT2=[]
totalTimeT3=[]
spreadT1=[]
spreadT2=[]
spreadT3=[]
fftwPlanT1=[]
fftwPlanT2=[]
fftwPlanT3=[]
fftT1=[]
fftT2=[]
fftT3=[]

for dim in dimensions:
    for ftype in types:
        for trial in n_trials:

            #execute the test for this set of parameters
            out =  subprocess.run(["./finufftGuru_test", str(trial), str(ftype), str(dim), 
                                  str(modes[3*(dim-1)]),str(modes[3*(dim-1)+1]),  str(modes[3*(dim-1)+2]), 
                                  str(M_srcpts), str(tolerance), str(debug)], 
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            strOut = out.stdout.decode() #convert bytes to string
            #print(strOut)

            
            #parse the output and syphon into data arrays

            decimalMatchString = "\d+\.?\d+"
            sciNotString = "(\d*.?\d*e-\d* s)"

            ###############################################################################
            #total time speedup
            lineMatch = re.search(r'=.*$',strOut)
            totalSpeedup = re.search(decimalMatchString, lineMatch.group(0))
            if(not totalSpeedup): #whole number speedup edge case
                totalSpeedup = re.search("\d+", lineMatch.group(0))
            totalSpeedup = float(totalSpeedup.group(0))
            #print(totalSpeedup)
            
            if(ftype == 1):
                totalTimeT1.append(totalSpeedup)
            elif(ftype == 2):
                totalTimeT2.append(totalSpeedup)
            else:
                totalTimeT3.append(totalSpeedup)

            ###############################################################################
                
            #spread (old) / [sort+spread]  (new)
            n_sorts = 1
            newSort = 0

            lineMatch = re.findall('.*setNUpoints.*sort.*',strOut)
            for match in lineMatch:
                sortVal = re.search(sciNotString, match)
                if(not sortVal):
                    sortVal = re.search(decimalMatchString, match)
                newSort = newSort + float(sortVal.group(0).split('s')[0].strip()) #trim off " s"
            
            #collect spreading if any
            newSpread=0
            lineMatch = re.search('.*finufft.*exec.*spread.*',strOut)
            if(lineMatch):
                spreadVal = re.search(sciNotString, lineMatch.group(0))
                if(not spreadVal):
                    spreadVal = re.search(decimalMatchString, lineMatch.group(0))
                newSpread = float(spreadVal.group(0).split('s')[0].strip())  #trim off " s"

            #collect interp if any
            newInterp=0
            lineMatch = re.search('.*finufft.*exec.*interp.*',strOut)
            if(lineMatch):
                interpVal = re.search(sciNotString, lineMatch.group(0))
                if(not interpVal):
                    interpVal = re.search(decimalMatchString, lineMatch.group(0))
                newInterp = float(interpVal.group(0).split('s')[0].strip())  #trim off " s"
            
            #collect the spread timings for each trial of old
            totalOldSpread=0   
            lineMatch = re.findall('.*spread.*ier.*', strOut) #gets spread AND unspread
            if(lineMatch):
                for match in lineMatch:
                    if(match):
                        oldSpreadVal = re.search(sciNotString, match)
                        if(not oldSpreadVal):
                            oldSpreadVal = re.search(decimalMatchString, match)
                        oldSpreadVal = oldSpreadVal.group(0).split('s')[0].strip() #trim off " s"
                        totalOldSpread = totalOldSpread + float(oldSpreadVal)
                                    
            spreadRatio = round((totalOldSpread)/(newSort + newSpread + newInterp),3)

            if(ftype == 1):
                spreadT1.append(spreadRatio)
            elif(ftype == 2):
                spreadT2.append(spreadRatio)
            else:
                spreadT3.append(spreadRatio)
            ###############################################################################

            #fftw_plan(old) / fftw_plan(new)

            planSciNotString = '(\(64\)[ \t]+)(\d*.?\d*e-\d* s)'
            planDecimalMatchString= '(\(64\)[ \t]+)(\d*\.?\d*)'
            
            #collect new fftw_plan time
            new_fftwPlan=0
            lineMatch = re.search(".*make plan.*fftw plan \(64\).*",strOut)
            if(lineMatch):
                fftwPlanVal = re.search(planSciNotString, lineMatch.group(0))
                if(not fftwPlanVal):
                    fftwPlanVal = re.search(planDecimalMatchString, lineMatch.group(0))
                new_fftwPlan = float(fftwPlanVal.group(2).split('s')[0])  #strip off s

            #collect the fftw_plan timings for each trial of old
            totalOldfftwPlan=0   
            lineMatch = re.findall('(?<!\(make plan\))fftw plan \(64\).*', strOut) #all fftw plan lines that don't include "make plan"
            if(lineMatch):
                for match in lineMatch:
                    if(match):
                        oldfftwPlanVal = re.search(planSciNotString, match)
                        if(not oldfftwPlanVal):
                            oldfftwPlanVal = re.search(planDecimalMatchString, match)
                        oldfftwPlanVal = float(oldfftwPlanVal.group(2).split('s')[0]) #trim off " s"
                        totalOldfftwPlan = totalOldfftwPlan + oldfftwPlanVal
            
            fftwPlanRatio = round(totalOldfftwPlan/new_fftwPlan,3)
            
            if(ftype == 1):
                fftwPlanT1.append(fftwPlanRatio)
            elif(ftype == 2):
                fftwPlanT2.append(fftwPlanRatio)
            else:
                fftwPlanT3.append(fftwPlanRatio)
            
            
            ###############################################################################

            #fftw_exec(old) / fftw_exec(new)

            #collect new fft time
            new_fft=0
            lineMatch = re.search(".*finufft_exec.*fft.*",strOut)
            if(lineMatch):
                fftVal = re.search(sciNotString, lineMatch.group(0))
                if(not fftVal):
                    fftVal = re.search(decimalMatchString, lineMatch.group(0))
                new_fft = float(fftVal.group(0).split('s')[0])  #strip off s

            #collect the fftw_plan timings for each trial of old
            totalOldfft=0   
            lineMatch = re.findall("(.*fft \(\d+ threads\))(.*)", strOut) 
            if(lineMatch):
                for match in lineMatch:
                    if(match):
                        oldfftVal = re.search(sciNotString, match[1])
                        if(not oldfftVal): #search failed
                            oldfftVal = re.search(decimalMatchString, match[1])
                        oldfftVal = float(oldfftVal.group(0).split('s')[0]) #trim off " s"
                        totalOldfft = totalOldfft + oldfftVal
            
            fftRatio = round(totalOldfft/new_fft,3)
            
            if(ftype == 1):
                fftT1.append(fftRatio)
            elif(ftype == 2):
                fftT2.append(fftRatio)
            else:
                fftT3.append(fftRatio)
            
            ###############################################################################

import numpy as np
data = np.zeros(len(totalTimeT1), dtype=[('totalT1','float'), ('totalT2','float'), ('totalT3','float'),
                                         ('spreadT1','float'), ('spreadT2','float'), ('spreadT3','float'),
                                         ('fftwPlanT1','float'), ('fftwPlanT2','float'), ('fftwPlanT3','float'),
                                         ('fftT1','float'), ('fftT2','float'), ('fftT3','float')])

data['totalT1'] = totalTimeT1
data['totalT2'] = totalTimeT2
data['totalT3'] = totalTimeT3

data['spreadT1'] = spreadT1
data['spreadT2'] = spreadT2
data['spreadT3'] = spreadT3

data['fftwPlanT1'] = fftwPlanT1
data['fftwPlanT2'] = fftwPlanT2
data['fftwPlanT3'] = fftwPlanT3

data['fftT1'] = fftT1
data['fftT2'] = fftT2
data['fftT3'] = fftT3



np.savetxt('timing.data', data)






