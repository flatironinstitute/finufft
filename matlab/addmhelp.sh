#!/bin/bash
# Flesh out .docsrc into .docexp using tags, then
# insert these comment doc help blocks into all *.m matlab codes.
# Overwrites all m-files - be careful.
# Barnett 6/11/20

# stage 1: flesh out *.docsrc (input i) to *.docexp (output o)...
for i in *.docsrc
do
    o=${i/.docsrc/.docexp}
    #echo "$o"
    # create or overwrite output as 0-length file (not needed):  echo -n "" > $o
    while IFS= read -r line; do
        case $line in
            OPTS)
                # start of opts description, common to all routines
                cat opts.docbit
                ;;
            OPTS12)
                # just opts for types 1,2, not type 3
                cat opts12.docbit
                ;;
            TAIL)
                # errors and notes common to all routines
                cat tail.docbit
                ;;
            *)
                # all else is pipes through
                echo "$line"
                ;;
        esac
    done < $i > $o
    # (note sneaky use of pipes above, filters lines from $i, output to $o)
done

# stage 2: go through list of .m files and insert docs after functions...
for mfile in finufft?d?.m finufft_plan.m
do
    bash insertdoc.sh < $mfile > tmp
    # overwrites the .m file...
    mv -f tmp $mfile
done

# clean up
rm -f *.docexp
