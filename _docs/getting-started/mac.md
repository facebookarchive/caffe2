<block class="mac compile prebuilt" />

## Prerequisites

### Install Homebrew

Whenever possible, [Homebrew](http://brew.sh) is used to install dependencies.

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Install Python

[Python 2.7](https://www.python.org/download/releases/2.7/) is required to run Caffe2's Python modules. You can still use Caffe2's C++ libraries without Python if that is your preference. *MacOS X has Python built in by default*, and that can be used to run Caffe2. To check your version:

```
python --version
```

Caffe2 currently supports Python 2.7. We hope to support Python 3 in the near future. If you're already using Python 3, consider using virtualenv or Anaconda and install Caffe2 in a Python 2.7 environment.


> The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.

<block class="mac compile" />

### Install Git

While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.

```
brew install git
```

### Required Dependencies

1. Xcode Command Line Tools or a comparable compiler is needed to build from source.

  - [Xcode](https://developer.apple.com/xcode/): Apple developer account required

2. (optional) GPU support - you don't need this if you are only using CPU mode

  - [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone): install from NVIDIA's site; free developer account required
  - [Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)

3. Other dependencies will be installed for you in /third_party in Caffe2's source. [more info](getting-started.html#whats-in-third-party)

> Make sure you have installed Xcode and CUDA (if using GPU) before running these

```
brew install glog automake protobuf leveldb lmdb
sudo pip install numpy protobuf
```

## Compilation

To compile Caffe2, first ensure the [prerequisites above]() are installed. Then you can download for compilation.

```
git clone --recursive https://github.com/caffe2/caffe2.git
cd caffe2
make
make install
```

## Verification

To verify that Caffe2 was installed correctly, run the following:

```
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

An output of `Success` means you are ready to with Caffe2 - congratulations!

An output of `Failure` usually means you have not installed one of the dependencies correctly.

<block class="mac prebuilt" />

## Prebuilt

** ADD LINK TO PREBUILT WHEEL HERE **

<block class="mac compile prebuilt" />

### Suggested Dependencies

Strictly speaking, you now have everything you need to run the core Caffe2 successfully. However, for real-world deep learning (e.g., image processing, mathematical operations, etc), there are other dependencies that you will want to install in order to experience the full features of Caffe2.

- [OpenCV](http://opencv.org/) for image-related operations.
- [OpenMPI](http://www.open-mpi.org/) for MPI-related Caffe2 operators.
- [RocksdB](http://rocksdb.org) for Caffe2's RocksDB IO backend.
- [ZeroMQ](http://zeromq.org/), needed for Caffe2's ZmqDB IO backend (serving data through a socket).
- [cuDNN](https://developer.nvidia.com/cudnn), needed for Caffe2's cuDNN operators.

The installation for OpenCV, OpenMPI and cuDNN are too detailed and complex to describe here, but you can follow the respective installation guides for each to install them.

You can install the others via [Homebrew](http://brew.sh):

```
brew install rocksdb, zeromq
```

There are also various Python libraries that will be valuable in your experience with Caffe2.

- [Flask](http://flask.pocoo.org/)
- [Jupyter](https://ipython.org/) for the Jupyter Notebook
- [Matplotlib](http://matplotlib.org/)
- [Pydot](https://pypi.python.org/pypi/pydot)
- [Python-nvd3](https://pypi.python.org/pypi/python-nvd3/)
- [SciPy](https://www.scipy.org/)
- [Tornado](http://www.tornadoweb.org/en/stable/)
- [Scikit-Image](http://scikit-image.org/)

```
sudo pip install flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image
```

### What's in Third Party?

- Android cmake
- Benchmark
- cnmem
- cub
- eigen
- googletest
- ios-cmake
- nccl
- nervanagpu
- NNPACK
- protobuf
- pybind11

<block class="mac docker" />

Mac docker images are currently not available. We are looking into providing them sometime in the future.
