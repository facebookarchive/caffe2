#!/bin/bash

set -ex

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please specify PYTHON_VERSION..."
  exit 1
fi

install_ubuntu_deadsnakes() {
  apt-get install -y --no-install-recommends software-properties-common
  add-apt-repository ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y --no-install-recommends "$1"
}

install_ubuntu() {
  apt-get update

  case "$PYTHON_VERSION" in
    2*)
      apt-get install -y --no-install-recommends \
              python-dev
      PYTHON=python2
      ;;
    3.5)
      apt-get install -y --no-install-recommends \
              python3-dev
      PYTHON=python3.5
      ;;
    3.6)
      install_ubuntu_deadsnakes python3.6-dev
      PYTHON=python3.6
      ;;
    3.7)
      install_ubuntu_deadsnakes python3.7-dev
      PYTHON=python3.7
      ;;
    *)
      echo "Invalid PYTHON_VERSION..."
      exit 1
      ;;
  esac

  # Clean up
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  source /etc/os-release
  if [ "$ID" != "centos" ]; then
    echo "Unknown ID: $ID"
    exit 1
  fi

  case "$PYTHON_VERSION" in
    2*)
      yum install -y \
          python-devel \
      PYTHON=python2
      ;;
    3.4)
      yum install -y \
          python34-devel \
      PYTHON=python3
      ;;
    *)
      echo "Invalid PYTHON_VERSION..."
      exit 1
      ;;
  esac

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi

# Install pip from source.
# The python-pip package on Ubuntu Trusty is old
# and upon install numpy doesn't use the binary
# distribution, and fails to compile it from source.
curl -O https://pypi.python.org/packages/11/b6/abcb525026a4be042b486df43905d6893fb04f05aac21c32c638e939e447/pip-9.0.1.tar.gz
tar zxf pip-9.0.1.tar.gz
pushd pip-9.0.1
"$PYTHON" setup.py install
popd
rm -rf pip-9.0.1*

# Install setuptools
# setuptools 38.5.2 seems to be buggy, see error in
# https://ci.pytorch.org/jenkins/job/caffe2-docker/job/py3.6-gcc5-ubuntu16.04/35/consoleFull
pip install -U pip setuptools!=38.5.2

# tornado 5.0 requires Python 2.7.9+ or 3.4+
if [[ $($PYTHON -c 'import sys; print(int(sys.version_info <= (2, 7, 9) or sys.version_info <= (3, 4)))' == 1) ]]; then
    pip install tornado<5.0
fi

# Need networkx 2.0 because bellmand_ford was moved in 2.1 . Scikit-image by
# defaults installs the most recent networkx version, so we install this lower
# version explicitly before scikit-image pulls it in as a dependency
pip install networkx==2.0

pip install --no-cache-dir \
    click \
    future \
    hypothesis \
    jupyter \
    numpy \
    protobuf \
    pytest \
    scipy==0.19.1 \
    scikit-image \
    tabulate \
    virtualenv
