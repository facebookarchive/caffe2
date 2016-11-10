FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  libeigen3-dev \
  libgoogle-glog-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libprotobuf-dev \
  libsnappy-dev \
  zlib1g-dev \
  libbz2-dev \
  protobuf-compiler \
  python-dev \
  python-pip \
  autoconf \
  automake \
  libtool \
  graphviz \
  libatlas-base-dev

# Caffe2 works best with openmpi 1.8.5 or above (which has cuda support).
# If you do not need openmpi, skip this step.
RUN OPENMPI_VERSION=1.10.3 && \
    wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-cuda --prefix=/usr --disable-getpwuid && \
    make -j"$(nproc)" install && \
    rm -rf /openmpi-${OPENMPI_VERSION}

# Caffe2 requires zeromq 4.0 or above, manually install.
# If you do not need zeromq, skip this step.
RUN mkdir /tmp/zeromq-build && \
  cd /tmp/zeromq-build && \
  wget https://github.com/zeromq/zeromq4-1/archive/v4.1.3.tar.gz && \
  tar xzvf v4.1.3.tar.gz --strip 1 && \
  ./autogen.sh && \
  ./configure --prefix=/usr --without-libsodium && \
  make -j"$(nproc)" && make -j"$(nproc)" install && \
  cd / && \
  rm -rf /tmp/zeromq-build

# pip self upgrade
RUN pip install --upgrade pip

# Python dependencies
RUN pip install \
  matplotlib \
  numpy \
  protobuf

################################################################################
# Step 3: install optional dependencies ("good to have" features)
################################################################################

RUN pip install \
  flask \
  ipython \
  notebook \
  pydot \
  python-nvd3 \
  scipy \
  tornado

# This is intentional. scikit-image has to be after scipy.
RUN pip install \
  scikit-image

################################################################################
# Step 4: set up caffe2
################################################################################

WORKDIR /workspace
COPY . .

# Get the repository, and build.
RUN make -j"$(nproc)"


