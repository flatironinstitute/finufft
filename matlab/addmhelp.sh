#!/bin/bash
# Main script to add in matlab doc help strings, in automated way that means
# duplicated info (opts, errors, etc) need appears only once, in .docsrc files.
# It expands .docsrc into .docexp using tags, then
# prepend these comment doc help blocks onto .m matlab codes.
# Warning: overwrites all m-files, stripping first % or empty lines; careful!
# Barnett 6/12/20, simplified.

# stage 1: flesh out *.docsrc (input i) to *.docexp (output o)...
for i in *.docsrc
do
    o=${i/.docsrc/.docexp}
    #echo "$o"
    # create or overwrite output as 0-length file (not needed):  echo -n "" > $o
    while IFS= read -r line; do
        # we define all tags here
        case $line in
            ISIGNEPS)
                # isign and eps descriptions common to all routines
                cat isigneps.docbit
                ;;
            OPTS)
                # opts descriptions common to all routines
                cat opts.docbit
                ;;
            OPTS12)
                # just opts for types 1,2, not type 3
                cat opts12.docbit
                ;;
            NOTES)
                # notes for the simple/many interfaces, not guru
                cat notes.docbit
                ;;
            *)
                # all else is piped through
                echo "$line"
                ;;
        esac
    done < $i > $o
    # (note sneaky use of pipes above, filters lines from $i, output to $o)
done

# stage 2: prepend doc to needed .m files after first stripping any top % block
for mfile in finufft?d?.m finufft_plan.m
do
    o=${mfile/.m/.docexp}
    # read from .m (skipping leading % or empty lines), appending to .docexp:
    reachedcode=
    while IFS= read -r line; do
        # in bash, a string is "true" iff nonzero length...
        if [[ $reachedcode || ($line && $line != %*) ]]; then
            echo "$line"
            # bash vars are strings, not numbers or bools, so put anything...
            reachedcode=yes
        fi
    done < $mfile >> $o
    # (note pipes above). Now overwrite that .m file...
    mv -f $o $mfile
done

# clean up
rm -f *.docexp

# debug note: to debug, best to echo "$stuff" 1>&2   so it goes to stderr.
