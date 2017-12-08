#!/bin/bash

set -ex

# This follows the instructions from
# https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile
# as closely as possible to install and build. Anaconda should already be
# installed.

# Required dependencies
apt-get update
apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libprotobuf-dev \
      protobuf-compiler \
      python-dev \
      python-pip
pip install numpy protobuf

# Optional dependencies
apt-get install -y --no-install-recommends libgflags-dev
apt-get install -y --no-install-recommends \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      openmpi-bin \
      openmpi-doc \
      python-pydot
pip install \
      flask \
      future \
      graphviz \
      hypothesis \
      jupyter \
      matplotlib \
      pydot python-nvd3 \
      pyyaml \
      requests \
      scikit-image \
      scipy \
      setuptools \
      six \
      tornado
