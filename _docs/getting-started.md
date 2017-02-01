---
docid: getting-started
title: Getting Started
layout: docs
permalink: /docs/getting-started.html
---

## Installing and Building Caffe2

[![Build Status](https://travis-ci.org/bwasti/caffe2.svg?branch=master)](https://travis-ci.org/bwasti/caffe2)

Thanks for your interesting in Caffe2! If you're new to Deep Learning you might want to take a quick look at our ["What is Deep Learning?"](index.html#caffe2-what-is-deep-learning) section first. If you already know about deep learning, but are new to Caffe you might want to take a look at our ["Why Use Caffe2?"](index.html#caffe2-why-use-caffe2) section before tackling the installation.

Once you've successfully installed Caffe2, check out our [Tutorials](tutorials.html) for a jumpstart on how to use Caffe2 for neural networking and deep learning, how Caffe2 can add deep learning your mobile application, or how Caffe2 can make large-scale distributed training possible for all of your deep learning scalability needs.

Ready to install Caffe2? Great! In order to install or try out Caffe2, you have several options:

- Pre-configured system images
  - [x] [Docker](#installing-and-building-caffe2-compilation-docker-support)
  - [ ] AWS (coming soon!)
- Compile it for your Operating System
  - [x] [MacOSx](#installing-and-building-caffe2-compilation-macosx)
  - [x] [Linux / Ubuntu](#installing-and-building-caffe2-compilation-ubuntu)
  - [ ] Windows (coming soon!)
- Mobile
  - [x] Android / Android Studio
  - [x] iOS / Xcode
- [ ] Pre-built binaries (coming soon!)

[Demos](index.html#caffe2-getting-started-with-caffe2-demos) are also a good option if you want to see it in action without setting it up yourself.

## Getting the Source

    git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2


If the recursive option doesn't work for your version of git, or if you already cloned the repo without downloading and initializing the submodules you can try the following steps. Definitely try this if you're getting errors trying to compile.


    git clone --recursive https://github.com/caffe2/caffe2.git
    cd caffe2
    git submodule init
    git submodule update


## Installation

  The Caffe2 library's dependency is largely similar to that of Caffe's. Thus, if you have installed Caffe in the past, you should most likely be good to go. Otherwise, please check the prerequisites and specific platforms' guides.

  For MacOSx, Caffe2 uses a Homebrew build script so that we can deal with multiple targets as well as optional dependencies. The format is similar to build systems like [Bazel](http://bazel.io) and [Buck](https://buckbuild.com/) with some custom flavors. It is based on python, so you will need to have Python installed.

  When updating Caffe2, it's best to `make clean` before re-compiling.

## Prerequisites

  Caffe2 has several dependencies.

  * A C++ compiler that supports C++11.
  * [CUDA](https://developer.nvidia.com/cuda-zone) is required for GPU mode.
      * library version above 6.5 are needed for C++11 support, and 7.0 is recommended.
  * `protobuf`, `glog`, `gflags`, `eigen3`

### Optional Dependencies

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

### Python Versions
  It is possible to compile and run Caffe2 with 2.7 up to 3.5 versions of Python as well as using Python environments provided by different Python distributions such as Anaconda. If you are on a Mac and trying this for the first time we first suggest that you try using the default version of Python before jumping to a distribution or using environments. This may simplify things dramatically.

  [Anaconda](https://store.continuum.io/cshop/anaconda/) Python distribution may be used as well, which provides most of the necessary packages, as well as the `hdf5` library dependency.

## Compilation

  Now that you have the prerequisites, you should be good to go. Caffe's build environment should be able to figure out everything itself. However, there may be cases when you need to manually specify some paths of the build toolchain. In that case, go to `build_env.py`, and modify the lines in the Env class (look for line `class Env(object)`) accordingly. Then, simply run

      make

  The build script should tell you what got built and what did not get built.

### MacOSx

#### OSX Prerequisites

  1. Install [Command Line Tools from Xcode](https://developer.apple.com/)
  2. Install [Homebrew](http://brew.sh/)

  Fetch the [latest source](#installing-and-building-caffe2-getting-the-source) code from Github if you haven't already.

  Several prerequisites are now installed via brew.   
  Note, installation might be able to just use automake as eigen is default, many of the "prerequisites" are now in third party, and the others were optional:


    brew install automake


  The previously known working install method used:


    brew install glog automake protobuf lmdb opencv libtool
    brew install homebrew/science/openblas


  Assuming everything above installs without errors you can move on to the make steps. Warnings should be fine and you can move ahead without trouble.

#### OSX Compilation

  If you're starting from scratch, the commands below will create your */build* directory and begin the compilation process. Another directory will be created in your Caffe2 root directory called */install*. The cmake step uses the install directory and also turns off LevelDB. If you're not starting from scratch then delete your */build* and */install* folders first, then run the commands below.


    mkdir build && mkdir install && cd build
    cmake .. -DCMAKE_INSTALL_PATH=../install -DUSE_LEVELDB=OFF
    make


  Once the build completes without errors, you will want to:
    - [Configure Python](#installing-and-building-caffe2-compilation-configure-python)
    - [Test Caffe2 in Python](#installing-and-building-caffe2-compilation-test-caffe2)
    - [Install Tutorial Prerequisites](#installing-and-building-caffe2-compilation-tutorials-prerequisites)

  [Original Caffe's OSX guide](http://caffe.berkeleyvision.org/install_osx.html)

### Ubuntu

  For ubuntu 14.04 users, the Docker script may be a good example on the steps of building Caffe2. Please check `contrib/docker-ubuntu-14.04/Dockerfile` for details. For ubuntu 12.04, use `contrib/docker-ubuntu-12.04/Dockerfile`.

  Since most Ubuntu users will want to use Caffe2 for training, research, and a variety of testing we're throwing in the kitchen sink for options during this installation. You could certainly prune many of these packages if you wanted a leaner installation.

  If you're using a VM or a cloud solution, make sure you give yourself enough room for the compilation process. You will need at least 12 GB of disk space to get through all of the compilation. If you don't plan on using GPU, then you could skip the CUDA steps and use a much smaller disk image.

      sudo apt-get install libprotobuf-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev libgtest-dev liblmdb-dev libleveldb-dev libsnappy-dev python-dev python-pip libiomp-dev libopencv-dev libpthread-stubs0-dev cmake
      sudo pip install numpy
      wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
      sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
      sudo apt-get update
      sudo apt-get install cuda
      sudo apt-get install git
      sudo pip install protobuf

  Fetch the [latest source](#installing-and-building-caffe2-getting-the-source) code from Github if you haven't already.

      CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz" &&
      curl -fsSL ${CUDNN_URL} -O &&
      sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local &&
      rm cudnn-8.0-linux-x64-v5.1.tgz &&
      sudo ldconfig

      mkdir build && cd build
      cmake ..
      make  

  Once the build completes without errors, you will want to:
    - [Configure Python](#installing-and-building-caffe2-compilation-configure-python)
    - [Test Caffe2 in Python](#installing-and-building-caffe2-compilation-test-caffe2)
    - [Install Tutorial Prerequisites](#installing-and-building-caffe2-compilation-tutorials-prerequisites)

#### GPU Support

  Copy the nccl shared library to /usr/local/lib if you want GPU support.

### Configure Python

  You will need to update your PYTHONPATH environment variable to use the newly created files in your */install* directory. Update the directory in the command below to match your Caffe2 install folder path.


    sudo make install
    export PYTHOPATH=/usr/local


### Android

  To build for Android we recommend Android Studio. Other routes will work as well, but currently this documentation will only describe steps for this IDE.

  To build Caffe2 for Android run the following script:

    ./scripts/build_android.sh

### iOS

To build for iOS we recommend Xcode. Other routes will work as well, but currently this documentation will only describe steps for this IDE.

To build Caffe2 for iOS run the following script:

    ./scripts/build_ios.sh

### Test Caffe2

  To test if Caffe2 is working run the following:


    python -c 'from caffe2.python import core' 2>/dev/null && echo "Success!" || echo "uh oh!"


  If you get a result of "Success!" then you're ready to Caffe! If you get an "uh oh" then go back and check your console for errors and see if you missed anything. Many times this can be related to Python environments and you'll want to make sure you're running Python that's registered with the Caffe2 modules.

### Tutorials Prerequisites

  If you plan to run the tutorials and the Jupyter notebooks you can get these package from your package manager of choice: apt-get, pip, or Anaconda's conda. Here are examples using pip.

    sudo pip install ipython
    sudo pip install notebook
    sudo pip install matplotlib
    sudo pip install graphviz

### Docker Support

  If you have docker installed on your machine, you may want to use the provided Docker build files for simpler set up. Please check the `contrib/docker*` folders for details.

  Running these Docker images with CUDA GPUs is currently only supported on Linux hosts, as far as I can tell. You will need to make sure that your host driver is also 346.46, and you will need to invoke docker with

      docker run -t -i --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia0:/dev/nvidia0 [other cuda cards] ...

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
