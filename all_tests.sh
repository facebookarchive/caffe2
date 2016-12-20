#!/bin/bash

declare -i RESULT=0

for i in `find . -name \*test\*.py`; do
  python $i;
  RESULT+=$?;
done;

exit $RESULT

