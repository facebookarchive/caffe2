#!/bin/bash

# NOTE: All parameters to this function are forwared directly to conda-build
# and so will never be seen by the build.sh

set -ex

# portable_sed: A wrapper around sed that works on both mac and linux, used to
# alter conda-build files such as the meta.yaml
portable_sed () {
  if [ -z "$1" ]; then
    echo "Programmer error: No regex passed to portable_sed"
    exit 1
  fi
  if [ -z "$2" ]; then
    echo "Programmer error: No file passed to portable_sed"
    exit 1
  fi
  if [ "$(uname)" == 'Darwin' ]; then
    sed -i '' "$1" "$2"
  else
    sed -i "$1" "$2"
  fi
}

# enforce_version: Takes a package name and a version and finagles the
# meta.yaml to ask for that version specifically. If the package was specified 
# with a different version in the meta.yaml (or unspecified) then the
# specification will be changed to be exactly the given version
# TODO make this work when the package wasn't in meta.yaml to start with
# NOTE: this assumes that $META_YAML has already been set
enforce_version () {
  if [ -z "$1" ]; then
    echo "Programmer error: No package name passed to enforce_version"
    exit 1
  fi
  if [ -z "$2" ]; then
    echo "Programmer error: No version string passed to enforce_version"
    exit 1
  fi
  portable_sed "s/- ${1}.*/- ${1} ==${2}/" "${META_YAML}"
}

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
CONDA_BUILD_ARGS=()
CMAKE_BUILD_ARGS=()

# Build for Python 3.6
# Specifically 3.6 because the latest Anaconda version is 3.6, and so it's site
# packages have 3.6 in the name
PYTHON_FULL_VERSION="$(python --version 2>&1)"
if [[ "$PYTHON_FULL_VERSION" == *3.6* ]]; then
  CONDA_BUILD_ARGS+=(" --python 3.6")
fi

# Reinitialize submodules
git submodule update --init

# Pick correct conda-build folder
CAFFE2_CONDA_BUILD_DIR="${CAFFE2_ROOT}/conda"
if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda_full"
elif [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/cuda"
else
  CAFFE2_CONDA_BUILD_DIR="${CAFFE2_CONDA_BUILD_DIR}/no_cuda"
fi
META_YAML="${CAFFE2_CONDA_BUILD_DIR}/meta.yaml"

# Change the package name for CUDA builds to have the specific CUDA and cuDNN
# version in them
CAFFE2_PACKAGE_NAME="caffe2"
if [[ "${BUILD_ENVIRONMENT}" == *cuda* ]]; then
  # Build name of package
  CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-cuda${CAFFE2_CUDA_VERSION}-cudnn${CAFFE2_CUDNN_VERSION}"
  if [[ "${BUILD_ENVIRONMENT}" == *full* ]]; then
    CAFFE2_PACKAGE_NAME="${CAFFE2_PACKAGE_NAME}-full"
  fi

  # CUDA 9.0 and 9.1 are not in conda, and cuDNN is not in conda, so instead of
  # pinning CUDA and cuDNN versions in the conda_build_config and then setting
  # the package name in meta.yaml based off of these values, we let Caffe2
  # take the CUDA and cuDNN versions that it finds in the build environment,
  # and manually set the package name ourself.
  # WARNING: This does not work on mac.
  sed -i "s/caffe2-cuda\$/${CAFFE2_PACKAGE_NAME}/" "${META_YAML}"
fi

# If skipping tests, remove the test related lines from the meta.yaml and don't
# upload to Anaconda.org
if [ -n "$SKIP_CONDA_TESTS" ]; then
  portable_sed '/test:/d' "${META_YAML}"
  portable_sed '/imports:/d' "${META_YAML}"
  portable_sed '/caffe2.python.core/d' "${META_YAML}"

elif [ -n "$UPLOAD_TO_CONDA" ]; then
  # Upload to Anaconda.org if needed. This is only allowed if testing is
  # enabled
  CONDA_BUILD_ARGS+=(" --user ${ANACONDA_USERNAME}")
  CONDA_BUILD_ARGS+=(" --token ${CAFFE2_ANACONDA_ORG_ACCESS_TOKEN}")
fi

# Change flags based on target gcc ABI
if [ "$(uname)" == 'Darwin' -a -z "$USE_OLD_GCC_ABI" ]; then
  CMAKE_BUILD_ARGS+=("-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
  # Default conda channels use gcc 7.2 (for recent packages), conda-forge uses
  # gcc 4.8.5
  CONDA_BUILD_ARGS+=(" -c conda-forge")

  # gflags 2.2.1 uses the new ABI. Note this sed call won't work on mac
  # opencv 3.3.1 has a dependency on opencv_highgui that breaks
  enforce_version 'gflags' '2.2.0'
  enforce_version 'opencv' '3.3.0'
fi

# Build Caffe2 with conda-build
# If --user and --token are set, then this will also upload the built package
# to Anaconda.org, provided there were no failures and all the tests passed
CONDA_CMAKE_BUILD_ARGS="$CMAKE_BUILD_ARGS" conda build "${CAFFE2_CONDA_BUILD_DIR}" ${CONDA_BUILD_ARGS[@]} "$@"

# Install Caffe2 from the built package into the local conda environment
if [ -n "$CONDA_INSTALL_LOCALLY" ]; then
  conda install -y "${CAFFE2_PACKAGE_NAME}" --use-local
fi
