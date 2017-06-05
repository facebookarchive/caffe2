#!/bin/bash
# This script should be sourced, not executed
set -e

export BUILD_ANDROID=false
export BUILD_CUDA=false
export BUILD_GCC5=false
export BUILD_IOS=false
export BUILD_MKL=false

if [ "$BUILD" = 'linux' ]; then
    :
elif [ "$BUILD" = 'linux-gcc5' ]; then
    export BUILD_GCC5=true
elif [ "$BUILD" = 'linux-cuda' ]; then
    export BUILD_CUDA=true
elif [ "$BUILD" = 'linux-mkl' ]; then
    export BUILD_MKL=true
elif [ "$BUILD" = 'linux-android' ]; then
    export BUILD_ANDROID=true
elif [ "$BUILD" = 'osx' ]; then
    :
elif [ "$BUILD" = 'osx-ios' ]; then
    export BUILD_IOS=true
elif [ "$BUILD" = 'osx-android' ]; then
    export BUILD_ANDROID=true
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi
