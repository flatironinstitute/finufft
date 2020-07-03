#!/usr/bin/env python3

import math
import numpy as np
import re
import subprocess

import searchForTimeMetrics as stm

type = 1
n_trials = 1

#Small problem, sequential Multithreading
cmdStringOne_string = "./finufftGuru_test " + str(n_trials) + " "+ str(type) +" 3 1e1 1e1 1e1 1e4 1e-6 1 0" 
cmdStringOne_desc = "Small Problem, sequential Multithreaded"

#small problem, simulataneous single threaded, nested at end
cmdStringTwo_string = "./finufftGuru_test " + str(n_trials) + " "+ str(type) +" 3 1e1 1e1 1e1 1e4 1e-6 1 1" 
cmdStringTwo_desc = "Small Problem, single threaded sort in parallel"

#Large problem, sequential Multithreading
cmdStringThree_string = "./finufftGuru_test " + str(n_trials) + " "+ str(type) +" 3 1e2 1e2 1e2 1e7 1e-6 1 0" 
cmdStringThree_desc = "Large Problem, sequential Multithreaded"

#Large problem, simulataneous single threaded, nested at end
cmdStringFour_string = "./finufftGuru_test " + str(n_trials) + " "+ str(type) +" 3 1e2 1e2 1e2 1e7 1e-6 1 1" 
cmdStringFour_desc = "Large Problem, single threaded sort in parallel"

cmdStrings = [cmdStringOne_string, cmdStringTwo_string, cmdStringThree_string, cmdStringFour_string]
cmdDescs = [cmdStringOne_desc, cmdStringTwo_desc, cmdStringThree_desc, cmdStringFour_desc]

for i in range(len(cmdStrings)):
    cmdString = cmdStrings[i].split(' ')
    cmddesc = cmdDescs[i]
    print(cmddesc)
    print(cmdStrings[i])
    out =  subprocess.run([cmdString[0], cmdString[1], cmdString[2], cmdString[3],cmdString[4], cmdString[5],
                           cmdString[6], cmdString[7], cmdString[8], cmdString[8], cmdString[10]], 
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    strOut = out.stdout.decode() #convert bytes to string
    #print(strOut) #uncomment to see debug output
    ###############################################################################
    #Gather Spreading Stats
    
    #spread (old) / [sort+spread]  (new)            
    newSort = stm.sumAllTime('(.*finufft_setpts.*sort)(.*)', strOut) #if type three, adds together both sorts

    #collect spreading if any
    newSpread = stm.extractTime('(.*finufft.*exec.*spread)(.*)' , strOut) 

    #collect interp if any
    newInterp = stm.extractTime('(.*finufft.*exec.*interp)(.*)',strOut) 

    #collect the spread timings for each trial of old
    totalOldSpread = stm.sumAllTime('(.*spread.*ier)(.*)', strOut) #gets spread AND unspread (i.e. interpolation) for all trials

    spreadRatio = round(totalOldSpread/(newSort + newSpread + newInterp),5)

    print("Total New Sort/Spread/Interp: " )
    print(round(newSort + newSpread + newInterp,5))
    print("Total Old Sort/Spread/Interp: ")
    print(totalOldSpread)
    print("Spreading Ratio: Old/New " )
    print(spreadRatio)
    print("***************************************************************")
