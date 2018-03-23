#!/bin/bash

set -ex

# This install script is written for Linux systems
if [ "$(uname)" != "Linux" ]; then
  echo "Don't know how to build on $(uname)"
  exit 1
fi

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/.. && pwd)

# Setup sccache if SCCACHE_BUCKET is set
if [ -n "${SCCACHE_BUCKET}" ]; then
  mkdir -p ./sccache

  SCCACHE="$(which sccache)"
  if [ -z "${SCCACHE}" ]; then
    echo "Unable to find sccache..."
    exit 1
  fi

  # Setup wrapper scripts
  for compiler in cc c++ gcc g++ x86_64-linux-gnu-gcc; do
    (
      echo "#!/bin/sh"
      echo "exec $SCCACHE $(which $compiler) \"\$@\""
    ) > "./sccache/$compiler"
    chmod +x "./sccache/$compiler"
  done

  # CMake must find these wrapper scripts
  export PATH="$PWD/sccache:$PATH"
fi

# Setup ccache if configured to use it (and not sccache)
if [ -z "${SCCACHE}" ] && which ccache > /dev/null; then
  mkdir -p ./ccache
  ln -sf "$(which ccache)" ./ccache/cc
  ln -sf "$(which ccache)" ./ccache/c++
  ln -sf "$(which ccache)" ./ccache/gcc
  ln -sf "$(which ccache)" ./ccache/g++
  ln -sf "$(which ccache)" ./ccache/x86_64-linux-gnu-gcc
  export CCACHE_WRAPPER_DIR="$PWD/ccache"
  export PATH="$CCACHE_WRAPPER_DIR:$PATH"
fi

# Cmake flags for both Android and Linux
CMAKE_ARGS=("-DBUILD_BINARY=ON")
CMAKE_ARGS+=("-DUSE_OBSERVERS=ON")
CMAKE_ARGS+=("-DUSE_ZSTD=ON")

# Run a special script for Android or Anaconda
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  "${ROOT_DIR}/scripts/build_android.sh" ${CMAKE_ARGS[*]} "$@"
  exit 0
fi
if [[ "${BUILD_ENVIRONMENT}" == conda* ]]; then
  # click (required by onnx) wants these set
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8

  # SKIP_CONDA_TESTS refers to only the 'test' section of the meta.yaml
  export SKIP_CONDA_TESTS=1
  export CONDA_INSTALL_LOCALLY=1
  "${ROOT_DIR}/scripts/build_anaconda.sh" "$@"
  exit 0
fi

# Explicitly set Python executable.
# On Ubuntu 16.04 the default Python is still 2.7.
PYTHON="$(which python)"
PIP="$(which pip)"
if [[ "${BUILD_ENVIRONMENT}" == py3* ]]; then
  PYTHON=/usr/bin/python3
  PIP=/usr/local/bin/pip3
  CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=${PYTHON}")
fi

# This prefix corresponds to the default python installation, which can read
# user installed libraries in /usr/local/lib/pythonx.x/site-packages, even
# though the binary itself is at /usr/bin/python#
INSTALL_PREFIX="/usr/local"
CMAKE_ARGS+=("-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}")

# Install ONNX
"$PIP" install "${ROOT_DIR}/third_party/onnx"

case "${BUILD_ENVIRONMENT}" in
  *-mkl*)
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

    # Ensure FindCUDA.cmake can infer the right path to the CUDA toolkit.
    # Setting PATH to resolve to the right nvcc alone isn't enough.
    # See /usr/share/cmake-3.5/Modules/FindCUDA.cmake, block at line 589.
    export CUDA_PATH="/usr/local/cuda"

    # Ensure the ccache symlink can still find the real nvcc binary.
    export PATH="/usr/local/cuda/bin:$PATH"
    ;;
esac

# Try to include Redis support for Linux builds
CMAKE_ARGS+=("-DUSE_REDIS=ON")

# We test the presence of cmake3 (for platforms like Centos and Ubuntu 14.04)
# and use that if so.
if [[ -x "$(command -v cmake3)" ]]; then
    CMAKE_BINARY=cmake3
else
    CMAKE_BINARY=cmake
fi

# Run cmake from ./build directory
mkdir -p ./build
cd ./build

# Configure
${CMAKE_BINARY} "${ROOT_DIR}" ${CMAKE_ARGS[*]} "$@"

# Build
make "-j$(nproc)" install

# Now that we've installed Caffe2 libraries into /usr/local/lib, we will need
# to add that to LD_LIBRARY_PATH
echo "${INSTALL_PREFIX}/lib" | sudo tee /etc/ld.so.conf.d/caffe2.conf
sudo ldconfig
