#!/bin/bash
# Insert processed comment doc help blocks into all *.m matlab codes.
# Just a loop using insertdoc.sh
# Barnett 6/11/20

# list of .m file headers
for i in finufft1d1
    #finufft_plan
do
    bash insertdoc.sh < $i.m > tmp
# overwrites the .m file...
    mv -f tmp $i.m
done
rm -f tmp
