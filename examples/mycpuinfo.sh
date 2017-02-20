#!/bin/bash
# barnett 2/2/17

echo "what CPUs do I have?..."
grep "model name" /proc/cpuinfo | uniq
if hash lscpu 2> /dev/null; then   # only do it if cmd exists...
    lscpu
fi

#lscpu | egrep 'Thread|Core|Socket|^CPU\(|MHz'
# thanks to http://unix.stackexchange.com/questions/218074/how-to-know-number-of-cores-of-a-system-in-linux

# https://www.cyberciti.biz/faq/lscpu-command-find-out-cpu-architecture-information/
