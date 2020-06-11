#!/bin/bash
# This function reads text from stdin, and substitutes a block defined by
# start/end tag lines with the contents from a file, and outputs to stdout.
# 
# Usage:
#    replaceblkwithfile.sh STARTTAG ENDTAG FILENAME
#
# Example:
#    printf 'text\nwith\nmultiple\nlines\n' > tmp
#    seq 100 105 | ./replaceblkwithfile.sh '^102' '^104' tmp
# Example output:
#    100
#    101
#    text
#    with
#    multiple
#    lines
#    105
#
# Adapted from https://unix.stackexchange.com/questions/141387/sed-replace-string-with-file-contents

BLOCK_StartRegexp="${1}"
BLOCK_EndRegexp="${2}"
FILE="${3}"
sed -e "/${BLOCK_EndRegexp}/a ___tmpMark___" -e "/${BLOCK_StartRegexp}/,/${BLOCK_EndRegexp}/d" | sed -e "/___tmpMark___/r ${FILE}" -e '/___tmpMark___/d'
