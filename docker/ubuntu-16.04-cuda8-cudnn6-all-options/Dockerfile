FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER Aaron Markham <aaronmarkham@fb.com>

# caffe2 install with gpu support

########## REQUIRED DEPENDENCIES ################
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    git \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

########## INSTALLATION STEPS ###################
RUN git clone --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF
RUN make install \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
