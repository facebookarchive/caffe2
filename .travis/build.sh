#!/bin/bash

mkdir build
cd build

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#************#
# OS X build #
#************#
  cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON -DUSE_OPENCV=off && make -j 4
else
#*************#
# Linux build #
#*************#
  cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON && make -j 4
fi
