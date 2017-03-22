{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

See the Troubleshooting section below for tips.

### Required Dependencies

[Anaconda](https://www.continuum.io/downloads). Python 2.7 version is needed for Caffe2.

[Homebrew](https://brew.sh/).

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
automake \
graphviz \
hypothesis \
leveldb \
lmdb \
zeromq
```

## Options that get installed already with other packages

flask (already installed)
jupyter (comes with anaconda)
matplotlib (probably can skip as it comes with numpy)
pydot (not found, found pydot-ng)
pyyaml (already installed)
scikit-image (probably can skip - comes with numpy)
scipy (probably can skip - comes with numpy)
setuptools (already installed)
tornado (already installed)


### Clone & Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make && cd build && sudo make install
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

### Python Configuration

You will want to add the build folder to your PYTHONPATH. `echo $PYTHONPATH` and if it's not in there add `export PYTHONPATH=~/caffe2/build:$PATH` to .zshrc or whatever you're using. Modify the path according to where your Caffe2 build folder is.

### Troubleshooting

|Python errors
----|-----
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require [Python2.7](https://www.python.org/download/releases/2.7/). MacOSx Sierra comes pre-installed with Python 2.7.10, but you may need to update to run Caffe2. To check your version: `python --version`
Solution | You can install the package for Python: `brew install python`
Python environment | You may have another version of Python installed or need to support Python version 3 for other projects.
Solution | Try virtualenv or Anaconda. The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.
pip version | If you plan to use Python with Caffe2 then you need pip or Anaconda to install packages.
Solution | pip comes along with `brew install python`

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
Solution | Make sure you've installed protobuf in **both** of these two ways: `brew install protbuf && sudo pip install protobuf`
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

- [Download the prebuilt Caffe2 Python Wheel](** ADD LINK TO PREBUILT WHEEL HERE **)
- Then run:

```bash
pip install caffe2.whl
```

This will also install various [third party](#whats-in-third-party) tools as well.

<block class="mac docker" />

<block class="mac cloud" />
