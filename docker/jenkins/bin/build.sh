#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../../.. && pwd)

# Run build script from scripts if applicable
if [[ "${BUILD_ENVIRONMENT}" == *android ]]; then
  export ANDROID_NDK=/opt/ndk
  ./scripts/build_android.sh "$@"
  exit 0
fi

# Run cmake from ./build directory
mkdir -p ./build
cd ./build

# Setup ccache symlinks
if which ccache > /dev/null; then
  mkdir -p /tmp/ccache
  pushd /tmp/ccache
  ln -sf "$(which ccache)" cc
  ln -sf "$(which ccache)" c++
  ln -sf "$(which ccache)" gcc
  ln -sf "$(which ccache)" g++
  ln -sf "$(which ccache)" nvcc
  export PATH=$PWD:$PATH
  popd
fi

CMAKE_ARGS=("-DCMAKE_INSTALL_PREFIX=/usr/local/caffe2")

case "${BUILD_ENVIRONMENT}" in
  *-mkl)
    CMAKE_ARGS+=("-DBLAS=MKL")
    ;;
  *-cuda*)
    CMAKE_ARGS+=("-DUSE_CUDA=ON")
    CMAKE_ARGS+=("-DCUDA_ARCH_NAME=Maxwell")
    CMAKE_ARGS+=("-DUSE_NNPACK=OFF")

    # Explicitly set path to NVCC such that the symlink to ccache is used
    CMAKE_ARGS+=("-DCUDA_NVCC_EXECUTABLE=/tmp/ccache/nvcc")

    # The CMake code that finds the CUDA distribution looks for nvcc in $PATH.
    # This doesn't work here, as that would yield /tmp/ccache, which doesn't
    # contain a CUDA distribution. We need it to resolve to /usr/local/cuda/bin,
    # so we add it to $PATH here. We can make CMake still use ccache by
    # specifying CUDA_NVCC_EXECUTABLE above.
    export PATH=/usr/local/cuda/bin:$PATH
    ;;
esac

# Configure
cmake .. ${CMAKE_ARGS[*]} "$@"

# Build
if [ "$(uname)" == "Linux" ]; then
  make "-j$(nproc)" install
else
  echo "Don't know how to build on $(uname)"
  exit 1
fi
