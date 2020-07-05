#!/bin/bash
# Output number of logical cores as a string, OS-indep.  Barnett 7/5/20.

# Linux and OSX for now. this doesn't handle non-linux unices.

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
        lscpu -p | egrep -v '^#' | wc -l
	;;
    Darwin*)
        sysctl -n hw.logicalcpu_max
	;;
    MINGW*)
        # not sure this correct...
        echo "$NUMBER_OF_PROCESSORS"
        ;;
    *)        
	echo "I'm in an unknown or unsupported operating system: ${unameOut}" >&2
        ;;
esac
