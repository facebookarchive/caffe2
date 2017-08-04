FROM ubuntu:16.04
MAINTAINER Aaron Markham <aaronmarkham@fb.com>

# caffe2 install with cpu support

########## REQUIRED DEPENDENCIES ################
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    python-pip \
    protobuf-compiler \
    python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir future hypothesis numpy protobuf six

########## INSTALLATION STEPS ###################
RUN git clone --branch v0.8.0 --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make install \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
