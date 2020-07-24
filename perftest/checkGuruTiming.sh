#!/bin/bash
# by Andrea Malleo, summer 2019.

srcpts=1e7
tolerance=1e-6
debug=1
modes[0]=1e6
modes[1]=1
modes[2]=1
modes[3]=1e3
modes[4]=1e3
modes[5]=1
modes[6]=1e2
modes[7]=1e2
modes[8]=1e2

for dimension in 1 2 3
do
    for type in 1 2 3
    do
	for n_trials in 1 20 41
	do
	    declare -i row
	    row=${dimension}-1

	    declare -i index
	    index=row*3

	    declare -i modeNum
	    modeNum1=${modes[index]}
	    modeNum2=${modes[index+1]}
	    modeNum3=${modes[index+2]}

	    echo "./guru_timing_test ${n_trials} ${type} ${dimension} ${modeNum1} ${modeNum2} ${modeNum3} ${srcpts} ${tolerance} ${debug}"				 
	    ./guru_timing_test ${n_trials} ${type} ${dimension} ${modeNum1} ${modeNum2} ${modeNum3} ${srcpts} ${tolerance} ${debug}
	done
    done
done

			       
