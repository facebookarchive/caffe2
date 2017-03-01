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

########## INSTALLATION STEPS ###################
RUN cd caffe2 && make && cd build && make install

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
