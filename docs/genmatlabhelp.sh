#!/bin/bash
# this extracts and concatenates MATLAB documentation blocks from the .m files.
# Three steps: 1) keep comment lines beginning with %, 2) remove the first
# char of each line, then 3) remove the header (until the first line of ----).
# This relies on having such a -------- line in the .mw file.
# Barnett 11/2/17

# zero the size...
> matlabhelp.raw

# dump the matlab comment blocks
for i in ../matlab/finufft?d?.m
do
    printf "::\n\n" >> matlabhelp.raw
    sed -n '/^%/p' $i | sed 's/^.//' >> matlabhelp.raw
    printf "\n" >> matlabhelp.raw
done

