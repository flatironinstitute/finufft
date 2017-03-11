#!/bin/bash
# main script to run validation tests for FINUFFT library. Barnett 3/11/17

TESTS=testutils
echo validating FINUFFT library...

for t in $TESTS; do
  echo
  echo $t :
  ./$t > $t.out
  ok=$?
  if [ $ok -eq 0 ]; then echo completed; else echo crashed; fi 
  # insert fancy relative error flags here, depending on each code...
  numdiff -q $t.out $t.refout
  ok=$?
  if [ $ok -eq 0 ]; then echo passed; else echo failed; fi
done
