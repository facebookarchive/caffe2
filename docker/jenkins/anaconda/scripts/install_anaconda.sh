#!/bin/bash

set -ex

# Adapted from https://hub.docker.com/r/continuumio/anaconda/~/dockerfile/
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Install needed packages
apt-get update --fix-missing
apt-get install -y wget \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        mercurial \
        subversion

# Install anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
case "$ANACONDA_VERSION" in
  2*)
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
  ;;
  3*)
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
  ;;
  *)
    echo "Invalid ANACONDA_VERSION..."
    exit 1
  ;;
esac
/bin/bash ~/anaconda.sh -b -p /opt/conda
rm ~/anaconda.sh

echo 'export PATH=/opt/conda/bin:$PATH' > ~/.bashrc
