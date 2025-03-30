#!/bin/bash
# Generate GPU docs for Matlab as .m files in this directory.
# Barnett 3/30/25

# flesh out *.docsrc (input i) to *.m (output o)...
for i in *.docsrc
do
    o=${i/.docsrc/.m}
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
