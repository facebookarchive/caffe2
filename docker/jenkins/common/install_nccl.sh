#!/bin/bash

set -ex

[ -n "$UBUNTU_VERSION" ]

# This doesn't actually use any NCCL version yet, it's just hardcoded to
# 2.1 for now

# The nvidia website has separate packages for CUDA 9.0 and 8.0, but they
# appear to point to the same URL, so we ignore this difference for now

# There are only NCCL packages for Ubuntu 16.04 and 14.04
if [[ "$UBUNTU_VERSION" == 16.04 ]]; then
  NCCL_UBUNTU_VER=ubuntu1604
  NCCL_DEB='nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb'
elif [[ "$UBUNTU_VERSION" == 14.04 ]]; then
  NCCL_UBUNTU_VER=ubuntu1404
  NCCL_DEB='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
else
  echo "There is no NCCL package for Ubuntu version ${UBUNTU_VERSION}."
  echo "    NCCL will not be installed."
fi

if [ -n "$NCCL_UBUNTU_VER" ]; then
  curl -LO "http://developer.download.nvidia.com/compute/machine-learning/repos/${NCCL_UBUNTU_VER}/x86_64/${NCCL_DEB}"
  dpkg -i "${NCCL_DEB}"
  apt update
  apt install libnccl2 libnccl-dev
fi
