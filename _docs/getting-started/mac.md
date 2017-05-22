{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

The Mac build works easiest with Anaconda. Always pull the latest from github, so you get any build fixes. See the Troubleshooting section below for tips.

### Required Dependencies

[Anaconda](https://www.continuum.io/downloads). Python 2.7 version is needed for Caffe2, and Anaconda is recommended. See below for a brew/pip install path instead of Anaconda.

[Homebrew](https://brew.sh/). Install Homebrew or use your favorite package manager to install the following dependencies:

```bash
brew install \
automake \
cmake \
git \
glog \
protobuf
```

```
conda install -y --channel https://conda.anaconda.org/conda-forge  \
gflags \
glog  \
numpy \
protobuf=3.2.0
```

### Optional GPU Support

In the instance that you have a NVIDIA supported GPU in your Mac, then you should visit the NVIDIA website for [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and install the provided binaries.

### Optional Dependencies

```
conda install -y \
--channel https://conda.anaconda.org/conda-forge \
graphviz \
hypothesis \
leveldb \
lmdb \
requests \
unzip \
zeromq
```

### Brew and Pip Install Path

Follow these instructions if you want to build Caffe2 without Anaconda. Make sure when you use pip that you're pointing to a specific version of Python or that you're using environments.

```bash
brew install \
automake \
cmake \
git \
glog \
protobuf \
python
```

```bash
sudo -H pip install \
flask \
glog \
jupyter \
matplotlib \
numpy \
protobuf \
pydot \
python-gflags \
pyyaml \
scikit-image \
scipy \
setuptools \
tornado
```

### Clone & Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
```

We're going to build without CUDA, using the `-DUSE_CUDA=OFF` flag, since it would be rare at this point for your Mac to have GPU card with CUDA support.

```
mkdir build && cd build
cmake -DUSE_CUDA=OFF ..
sudo make install
```

Now test Caffe2:

```
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

### Python Configuration

You might need setup a PYTHONPATH environment variable. `echo $PYTHONPATH` and if it's not in there add `export PYTHONPATH=~/usr/local` to `.zshrc`, `.bash_profile` or whatever you're using.

Change to a different folder and test Caffe2 again. If you are using Anaconda or had multiple versions of Python on your system the test may fail once out of the build folder. You will want to update the Python bindings:

```
sudo install_name_tool -change libpython2.7.dylib ~/anaconda/lib/libpython2.7.dylib /usr/local/caffe2/python/caffe2_pybind11_state.so
```

### Troubleshooting

|Python errors
----|-----
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require [Python2.7](https://www.python.org/download/releases/2.7/). MacOSx Sierra comes pre-installed with Python 2.7.10, but you may need to update to run Caffe2. To check your version: `python --version`
Solution | You can install the package for Python: `brew install python` or install [Anaconda](https://www.continuum.io/downloads).
Python environment | You may have another version of Python installed or need to support Python version 3 for other projects.
Solution | Try virtualenv or Anaconda. The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.
pip version | If you plan to use Python with Caffe2 then you need pip or Anaconda to install packages.
Solution | `pip` comes along with [Homebrew's python package](https://brew.sh/) or `conda` with [Anaconda](https://www.continuum.io/downloads).

|Building from source
----|-----
Anaconda | Test that your terminal is ready with `conda`. Make sure your PATH includes Anaconda.
Solution | `echo $PATH` and if it's not in there add `export PATH=/anaconda/bin:$PATH` to .zshrc or whatever you're using.)
Homebrew | Test that your terminal is ready with `brew install wget`.
Solution | Run this to install Homebrew: `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
OS version | Caffe2 is known to work on Sierra (others TBD after testing)
git | While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.
Solution | `brew install git`
protobuf | You may experience an error related to protobuf during the make step.
Solution | Make sure you've installed protobuf in **both** of these two ways: `brew install protobuf && sudo pip install protobuf` OR `brew install protobuf && conda install -y --channel https://conda.anaconda.org/conda-forge protobuf=3.2.0`
xcode | You may need to install [Xcode](https://developer.apple.com/xcode/) or at a minimum xcode command line tools.
Solution | You can install it via terminal using `xcode-select --install`

|GPU Support
----|-----
GPU | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for MacOSx.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone): install from NVIDIA's site; free developer account required. [NVIDIA MacOSx Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)

|Misc.
----|-----
malloc error | If you are using homebrew leveldb on a Mac OS, you might see an error warning you that malloc_zone_unregister() failed. This is not a caffe2 issue but is due to the homebrew leveldb having an incompatible memory allocator. It does not affect usage.

{{ outro | markdownify }}

<block class="mac prebuilt" />

### Prebuilt Caffe2 Python Wheel

This installer is in testing. It is expecting that you're running Python 2.7.13.

[https://s3.amazonaws.com/caffe2/installers/Caffe2-0.7.0-cp27-cp27m-macosx_10_12_x86_64.whl](https://s3.amazonaws.com/caffe2/installers/Caffe2-0.7.0-cp27-cp27m-macosx_10_12_x86_64.whl)

```bash
pip install Caffe2-0.7.0-cp27-cp27m-macosx_10_12_x86_64.whl
```

<block class="mac docker" />

<block class="mac cloud" />
