#!/bin/bash
# Output number of physical cores as a string, OS-indep. Barnett 7/5/20.

# see:
# https://stackoverflow.com/questions/6481005/how-to-obtain-the-number-of-cpus-cores-in-linux-from-the-command-line
# https://en.wikipedia.org/wiki/Uname

# Linux and MAX only. this doesn't handle non-linux unices.

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
        lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l
	;;
    Darwin*)
#	sysctl -n machdep.cpu.core_count
        sysctl -n hw.physicalcpu_max
	;;
    MINGW*)
        # not sure this is correct...
        echo "$NUMBER_OF_PROCESSORS"
        ;;
    *)        
	echo "I'm in an unknown or unsupported operating system: ${unameOut}" >&2
        ;;
esac
