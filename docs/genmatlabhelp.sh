#!/bin/bash
# this extracts and concatenates MATLAB documentation blocks from the .m files.
# Three steps: 1) keep comment lines beginning with %, 2) remove the first
# char of each line, then 3) remove the header (until the first line of ----).
# This relies on having such a -------- line in the .mw file.
# Barnett 11/2/17, changed output name 7/24/20.

# The output is a text file...
OUT=matlabhelp.doc

# zero the size...
> $OUT

# dump the matlab comment blocks
for i in ../matlab/finufft?d?.m ../matlab/finufft_plan.m
do
    printf "::\n\n" >> $OUT
    sed -n '/^%/p' $i | sed 's/^.//' >> $OUT
    printf "\n" >> $OUT
done
