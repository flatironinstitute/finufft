#!/bin/bash
# Insert processed comment doc help blocks into all *.m matlab codes.
# Just a loop using insertdoc.sh, but overwrites all m-files - be careful.
# Barnett 6/11/20

# list of .m files
for mfile in finufft?d?.m finufft_plan.m
do
    bash insertdoc.sh < $mfile > tmp
# overwrites the .m file...
    mv -f tmp $mfile
done
