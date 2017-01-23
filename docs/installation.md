# Installing and Building Caffe2

[![Build Status](https://travis-ci.org/bwasti/caffe2.svg?branch=master)](https://travis-ci.org/bwasti/caffe2)

- [MacOSx](#macosx)
- [Ubuntu](#Ubuntu)
- [Docker Support](#docker-support)
- [Tutorials & Python Support](#python)

## Getting the Source

```
    git clone https://github.com/bwasti/caffe2.git
    cd caffe2
```

## Installation

  The Caffe2 library's dependency is largely similar to that of Caffe's. Thus, if you have installed Caffe in the past, you should most likely be good to go. Otherwise, please check the prerequisites and specific platforms' guides.

  Caffe2 uses a Homebrew build script so that we can deal with multiple targets as well as optional dependencies. The format is similar to build systems like [Bazel](http://bazel.io) and [Buck](https://buckbuild.com/) with some custom flavors. It is based on python, so you will need to have Python installed.

  ## Developer Installation Notes (WIP)
  As there are many paths to installing and working with Caffe, you might find that a different route works best for you and your current development environment.
  These are the [Latest build notes](https://github.com/bwasti/caffe2/blob/master/README.md).

  - [Prerequisites](#prerequisites)
  - [Compilation](#compilation)
  - [Docker](#docker)
  - Platforms: [Ubuntu](#ubuntu), [OS X](OSX)

  When updating Caffe2, it's best to `make clean` before re-compiling.

## Prerequisites

  Caffe2 has several dependencies.

  * A C++ compiler that supports C++11.
  * [CUDA](https://developer.nvidia.com/cuda-zone) is required for GPU mode.
      * library version above 6.5 are needed for C++11 support, and 7.0 is recommended.
  * `protobuf`, `glog`, `gflags`, `eigen3`

  In addition, Caffe2 has several optional dependencies: not having these will not cause problems, but some components will not work. Note that strictly speaking, CUDA is also an optional dependency. You can compile a purely CPU-based Caffe2 by not having CUDA. However, since CUDA is critical in achieving high-performance computation, you may want to consider it a necessary dependency.

  * [OpenCV](http://opencv.org/), which is needed for image-related operations. If you work with images, you most likely want this.
  * [OpenMPI](http://www.open-mpi.org/), needed for MPI-related Caffe2 operators.
  * `leveldb`, needed for Caffe2's LevelDB IO backend. LevelDB also depends on `snappy`.
  * `rocksdb`, needed for Caffe2's RocksDB IO backend. RocksDB also depends on `snappy`, `bzip2`, and `zlib`.
  * `lmdb`, needed for Caffe2's LMDB IO backend.
  * [ZeroMQ](http://zeromq.org/), needed for Caffe2's ZmqDB IO backend (serving data through a socket).
  * [cuDNN](https://developer.nvidia.com/cudnn), needed for Caffe2's cuDNN operators.

  If you do not install some of the dependencies, when you compile Caffe2, you will receive warning message that some components are not correctly built. This is fine - you will be able to use the other components without problem.

  Pycaffe2 has its own natural needs, mostly on the Python side: `numpy (>= 1.7)` and `protobuf` are needed. We also recommend installing the following packages: `flask`, `ipython`, `matplotlib`, `notebook`, `pydot`, `python-nvd3`, `scipy`, `tornado`, and `scikit-image`.

  We suggest first installing the [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution, which provides most of the necessary packages, as well as the `hdf5` library dependency.

## Compilation

  Now that you have the prerequisites, you should be good to go. Caffe's build environment should be able to figure out everything itself. However, there may be cases when you need to manually specify some paths of the build toolchain. In that case, go to `build_env.py`, and modify the lines in the Env class (look for line `class Env(object)`) accordingly. Then, simply run

      make

  The build script should tell you what got built and what did not get built.

## MacOSx

### OSX Prerequisites

  1. Install [Command Line Tools from Xcode](https://developer.apple.com/)
  2. Install [Homebrew](http://brew.sh/)

  Fetch the latest source code from Github if you haven't already.

  ```
  git clone --recursive https://github.com/caffe2/caffe2.git
  cd caffe2
  ```
  If the recursive flag doesn't work for your version of git you can try the following, otherwise move on to the brew install step.

  ```
  git submodule init
  git submodule update
  ```

  Several prerequisites are now installed via brew.

  ```
  brew install glog automake protobuf leveldb lmdb opencv libtool
  brew install homebrew/science/openblas
  ```

  Assuming everything above installs without errors you can move on to the make steps. Warnings should be fine and you can move ahead without trouble.

  If you're starting from scratch then create your build directory and begin the make process.

  ```
  mkdir build && cd build
  cmake .. -DBLAS=OpenBLAS
  make
  ```

  [Original Caffe's OSX guide](http://caffe.berkeleyvision.org/install_osx.html)

## Ubuntu

### Docker + Ubuntu

  For ubuntu 14.04 users, the Docker script may be a good example on the steps of building Caffe2. Please check `contrib/docker-ubuntu-14.04/Dockerfile` for details. For ubuntu 12.04, use `contrib/docker-ubuntu-12.04/Dockerfile`.

```
  sudo apt-get install libprotobuf-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev libgtest-dev liblmdb-dev libleveldb-dev libsnappy-dev python-dev python-pip libiomp-dev libopencv-dev libpthread-stubs0-dev cmake
  sudo pip install numpy
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda
  sudo apt-get install git
```

```
  CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz" &&
  curl -fsSL ${CUDNN_URL} -O &&
  sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local &&
  rm cudnn-8.0-linux-x64-v5.1.tgz &&
  sudo ldconfig

  mkdir build && cd build
  cmake ..
  make
```

## Docker Support

  If you have docker installed on your machine, you may want to use the provided Docker build files for simpler set up. Please check the `contrib/docker*` folders for details.

  Running these Docker images with CUDA GPUs is currently only supported on Linux hosts, as far as I can tell. You will need to make sure that your host driver is also 346.46, and you will need to invoke docker with

      docker run -t -i --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia0 [other cuda cards] ...


## Tutorials Setup & Python Requirements

  To run the tutorials you'll need ipython-notebooks and matplotlib, which can be installed on OS X with:

  ```
      brew install matplotlib --with-python3
      pip install ipython notebook
  ```

## Build status (known working)

Ubuntu 14.04 (GCC)
- [ ] Default CPU build
- [x] Default GPU build

OS X (Clang)
- [x] Default CPU build
- [ ] Default GPU build

Options (both Clang and GCC)
- [ ] Nervana GPU
- [ ] ZMQ
- [ ] RocksDB
- [ ] MPI
- [ ] OpenMP
- [ ] No LMDB
- [ ] No LevelDB
- [x] No OpenCV

BLAS
- [x] OpenBLAS
- [x] ATLAS
- [ ] MKL

Other
- [ ] CMake 2.8 support
- [ ] List of dependencies for Ubuntu 14.04
- [ ] List of dependencies for OS X
