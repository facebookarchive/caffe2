FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER Aaron Markham <aaronmarkham@fb.com>

# caffe2 install with gpu support

########## REQUIRED DEPENDENCIES ################
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    git \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev

RUN git clone --recursive https://github.com/caffe2/caffe2.git && \
    pip install --upgrade pip setuptools wheel && \
    pip install future hypothesis numpy protobuf six

########## OPTIONAL DEPENDENCIES ################

RUN apt-get install -y --no-install-recommends \
    libatlas-base-dev  \
    libeigen3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python-dev \
    python-lmdb \
    python-matplotlib \
    python-networkx \
    python-numpy \
    python-pydot \
    python-setuptools \
    python-scipy \
    python-skimage \
    wget

RUN rm -rf /var/lib/apt/lists/*

RUN pip install \
    flask \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    tornado

########## INSTALLATION STEPS ###################
RUN cd caffe2 && mkdir build && cd build
WORKDIR /caffe2/build/
RUN cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF
RUN make install

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

########## TROUBLESHOOTING ######################
# this is added due to some issues that were found

# libdc1394 error
# for some reason once opencv is installed you may get errors with libdc1394
# http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394
# RUN ln /dev/null /dev/raw1394
# But that's not persistent http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu
# CMD sh -c 'ln -s /dev/null /dev/raw1394'
# or when instantiating the container use:
# --device /dev/null:/dev/raw1394
