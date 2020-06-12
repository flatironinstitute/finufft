#!/bin/bash
# Inserts expanded comment help string blocks for each function in
# a single MATLAB .m file, expected as stdin.
# Every time the input has a line starting with "function", the word starting
# the *next* non-empty line (which must be a comment) defines a filename (after
# conversion to lowercase and
# adding the .docsrc suffix) to process (flesh out by replacing blocks defined
# by start/end tokens), then insert. The old comment block is destroyed.
# To work, this relies on the first word of each *original* comment block in the
# in the input being
# the name of the command, which must therefore match the filename in *.docsrc
#
# test this script with (when finufft1d1.docexp is present):
# ./insertdoc.sh < finufft1d1.m
# and check it generates the right doc comments
#
# Barnett 6/11/20

# here the IFS=(blank) removes the list of usual separators just for this cmd,
# so whole line is read at once into variable $line...
while IFS= read -r line; do
    echo "$line"
    # split up (usual separators)...
    words=($line)
    # this will break if the MATLAB function def is split over >1 lines :(
    if [ "${words[0]}" == "function" ]; then
        # bad (since we're already inside a while loop!) hack to copy to next
        # *non-empty* line from stdin... (note IFS is usual spaces, etc)
        while read -r nextline; do
            # this replaces spaces by nothings, then -z checks if len=0...
            if [[ -z "${nextline// }" ]]; then
                echo "$nextline"
            else
                break
            fi
        done
        # nextline is now the above-mentioned non-empty line, so, split it...
        w=($nextline)
        if [ ${w[0]} != "%" ]; then
            echo "insertdoc.sh: function line must be followed by % comment line! Offending line:" 1>&2
            echo $nextline 1>&2
            exit 1
        fi
        # convert the 2nd word to lowercase via tr cmd; $(..) does cmd sub...
        filehead=$(echo ${w[1]} | tr A-Z a-z)
        i=$filehead.docexp
        # if this expanded doc file exsists, insert it and kill old % block...
        if [ -f "$i" ]; then
            # report progess to stderr not stdout...
            echo "insertdoc.sh: inserting $i ..." 1>&2
            # insert the file...
            cat "$i"
            # discard contiguous % lines, leaving stdin read pointer for next...
            while IFS= read -r nextline; do
                w=($nextline)
                if [ "${w[0]}" != "%" ]; then
                    echo "$nextline"
                    break
                fi
            done
        else
            echo "$nextline"
            echo "insertdoc.sh: $i not found (!); not replacing doc..." 1>&2
        fi
    fi
done
