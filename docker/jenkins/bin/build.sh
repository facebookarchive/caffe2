#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../../.. && pwd)

# Setup ccache symlinks
if which ccache > /dev/null; then
  mkdir -p ./ccache
  ln -sf "$(which ccache)" ./ccache/cc
  ln -sf "$(which ccache)" ./ccache/c++
  ln -sf "$(which ccache)" ./ccache/gcc
  ln -sf "$(which ccache)" ./ccache/g++
  export CCACHE_WRAPPER_DIR="$PWD/ccache"
  export PATH="$CCACHE_WRAPPER_DIR:$PATH"
fi

# Run build script from scripts if applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  ./scripts/build_android.sh "$@"
  exit 0
fi

# Run cmake from ./build directory
mkdir -p ./build
cd ./build

CMAKE_ARGS=("-DCMAKE_INSTALL_PREFIX=/usr/local/caffe2")

case "${BUILD_ENVIRONMENT}" in
  *-mkl)
    CMAKE_ARGS+=("-DBLAS=MKL")
    ;;
  *-cuda*)
    CMAKE_ARGS+=("-DUSE_CUDA=ON")
    CMAKE_ARGS+=("-DCUDA_ARCH_NAME=Maxwell")
    CMAKE_ARGS+=("-DUSE_NNPACK=OFF")

    # Add ccache symlink for nvcc
    ln -sf "$(which ccache)" "${CCACHE_WRAPPER_DIR}/nvcc"

    # Explicitly set path to NVCC such that the symlink to ccache is used
    CMAKE_ARGS+=("-DCUDA_NVCC_EXECUTABLE=${CCACHE_WRAPPER_DIR}/nvcc")

    # Ensure the ccache symlink can still find the real nvcc binary.
    export PATH="/usr/local/cuda/bin:$PATH"
    ;;
esac

# Configure
cmake "${ROOT_DIR}" ${CMAKE_ARGS[*]} "$@"

# Build
if [ "$(uname)" == "Linux" ]; then
  make "-j$(nproc)" install
else
  echo "Don't know how to build on $(uname)"
  exit 1
fi
