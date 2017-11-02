#!/bin/bash
# this cleans out just the MATLAB documentation blocks from the MWrap file
# Three steps: 1) keep comment lines beginning with %, 2) remove the first
# char of each line, then 3) remove the header (until the first line of ----).
# This relies on having such a -------- line in the .mw file.
# Barnett 11/2/17

sed -n '/^%/p' ../matlab/finufft.mw | sed 's/^.//' | sed -n '/-----/,$p' > matlabhelp.raw
