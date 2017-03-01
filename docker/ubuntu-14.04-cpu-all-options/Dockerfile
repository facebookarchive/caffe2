FROM ubuntu:14.04
MAINTAINER Aaron Markham <aaronmarkham@fb.com>

# caffe2 install with cpu support

########## REQUIRED DEPENDENCIES ################
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    git \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev

RUN git clone --recursive https://github.com/caffe2/caffe2.git && \
    pip install --upgrade pip && \
    pip install numpy protobuf 

########## OPTIONAL DEPENDENCIES ################

RUN apt-get install -y --no-install-recommends \
    libgflags-dev \
    libgoogle-glog-dev \
    libgflags2 \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python-numpy \
    python-setuptools \ 
    python-scipy \
    wget

RUN pip install setuptools flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml

########## INSTALLATION STEPS ###################
RUN cd caffe2 && make && cd build && make install

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
