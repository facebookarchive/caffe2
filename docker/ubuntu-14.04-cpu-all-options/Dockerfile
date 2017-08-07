FROM caffe2ai/caffe2:c2v0.8.0.cpu.min.ubuntu14.04
LABEL maintainer="aaronmarkham@fb.com"

# caffe2 install with cpu support
# 14.04 uses an older pip so using -U instead of --no-cache-dir

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgflags-dev \
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
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U \
    flask \
    graphviz \
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
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local
