{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

### Required Dependencies

```bash
brew install git glog automake protobuf
sudo pip2 install numpy protobuf
```

### Optional GPU Support

In the instance that you have a NVIDIA supported GPU in your Mac, then you should visit the NVIDIA website for [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and install the provided binaries.

### Optional Dependencies

```bash
brew install leveldb lmdb gflags rocksdb zeromq open-mpi opencv3 graphviz
sudo pip install flask graphviz hypothesis jupyter matplotlib pydot python-nvd3 pyyaml scikit-image scipy setuptools tornado
```

### Clone & Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

### Troubleshooting

|Python errors
----|-----
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require [Python2.7](https://www.python.org/download/releases/2.7/). MacOSx Sierra comes pre-installed with Python 2.7.10, but you may need to update to run Caffe2. To check your version: `python --version`
Solution | You can install the package for Python: `brew install python`
Python environment | You may have another version of Python installed or need to support Python version 3 for other projects.
Solution | Try virtualenv or Anaconda. The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.
pip version | If you plan to use Python with Caffe2 then you need pip.
Solution | pip comes along with `brew install python`

|Building from source
----|-----
OS version | Caffe2 is known to work on Sierra (others TBD after testing)
git | While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.
Solution | `brew install git`
protobuf | You may experience an error related to protobuf during the make step.
Solution | Make sure you've installed protobuf in **both** of these two ways: `brew install protbuf && sudo pip install protobuf`
brew | You need brew! Whenever possible, [Homebrew](http://brew.sh) is used to install dependencies. Use this to install it: `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
xcode | You need [Xcode](https://developer.apple.com/xcode/) or at a minimum xcode command line tools.
Solution | You can install it via terminal using `xcode-select --install`

| GPU Support
----|-----
GPU | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for MacOSx.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone): install from NVIDIA's site; free developer account required. [NVIDIA MacOSx Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)

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
