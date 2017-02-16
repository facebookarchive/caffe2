<block class="mac compile prebuilt" />

## Prerequisites

### Install Homebrew

Whenever possible, [Homebrew](http://brew.sh) is used to install dependencies.

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

### Install Python

[Python](https://www.python.org/) is core to run Caffe2. *MacOS X has Python built in by default*, and that can be used to run Caffe2. To check your version:

```
python --version
```

However, you can also install later versions of Python with Homebrew (e.g., version 3 instead of 2.7).

```
brew install python3
```

> The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.

<block class="mac compile" />

### Install Git

While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.

```
brew install git
```

### Required Dependencies

- [Xcode](https://developer.apple.com/xcode/)
- [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone)
- [C++ 11](https://en.wikipedia.org/wiki/C%2B%2B11)
- [Google Protocol Buffers](https://developers.google.com/protocol-buffers/)
- [Google Logging Module](https://github.com/google/glog)
- [gflags](https://gflags.github.io/gflags/)
- [Eigen 3](http://eigen.tuxfamily.org/)
- [NumPy](http://www.numpy.org/)

First, install Xcode from the [app store](https://itunes.apple.com/us/app/xcode/id497799835) and as described in the NVIDIA [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/).

At this point Xcode, CUDA and C++ 11 are installed. Now install the other dependencies.

```
## Make sure you have installed Xcode and CUDA before running these
brew install python3 protobuf glog gflags eigen
## Now install NumPy
sudo pip3 install numpy
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
- [LevelDB](http://leveldb.org/) for Caffe2's LevelDB IO backend.
- [RocksdB](http://rocksdb.org) for Caffe2's RocksDB IO backend.
- [LMDB](https://lmdb.readthedocs.io/en/release/) for Caffe2's LMDB IO backend.
- [ZeroMQ](http://zeromq.org/), needed for Caffe2's ZmqDB IO backend (serving data through a socket).
- [cuDNN](https://developer.nvidia.com/cudnn), needed for Caffe2's cuDNN operators.

The installation for OpenCV, OpenMPI and cuDNN are too detailed and complex to describe here, but you can follow the respective installation guides for each to install them.

You can install the others via [Homebrew](http://brew.sh):

```
brew install leveldb, rocksdb, lmdb, zeromq
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

<block class="mac docker" />

Mac docker images are currently not available. We are looking into providing them sometime in the future.
