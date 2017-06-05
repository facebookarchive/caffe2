#!/bin/bash
set -e
set -x

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

if [ "$BUILD_CUDA" = 'true' ]; then
    echo "Skipping tests for CUDA build."
    exit 0
fi
if [ "$BUILD_ANDROID" = 'true' ]; then
    echo "Skipping tests for Android build."
    exit 0
fi
if [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    echo "Skipping tests for OSX build."
    exit 0
fi

cd build
CTEST_OUTPUT_ON_FAILURE=1 make test
