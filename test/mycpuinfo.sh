#!/bin/bash
# Barnett 2/2/17
# Linux and OSX both, 11/1/18

echo "what CPUs do I have?..."
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
	echo "(I'm in a linux OS)"
	grep "model name" /proc/cpuinfo | uniq
	if hash lscpu 2> /dev/null; then   # only do it if cmd exists...
	    lscpu
	fi
	;;
    Darwin*)
	echo "(I'm in Mac OSX)"
	sysctl -n machdep.cpu.brand_string
	sysctl -a | grep machdep.cpu
	;;
    *)
	echo "I'm in an unknown or unsupported operating system";;
esac

# help from:

#lscpu | egrep 'Thread|Core|Socket|^CPU\(|MHz'
# thanks to http://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux

# https://www.cyberciti.biz/faq/lscpu-command-find-out-cpu-architecture-information/

# https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux/27776822
