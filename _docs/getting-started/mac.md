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
brew install leveldb lmdb gflags rocksdb zeromq open-mpi opencv3
sudo pip install setuptools flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml
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

### Docker Images

Docker images are currently in testing. If you would like to try them out, follow the below instructions.
Inside the [docker](../docker) folder are subfolders with a `Dockerfile` that contain the minimal dependencies and optional ones. You may remove specific optional dependencies if you wish. The folder's name describes the defaults that will be installed by that dockerfile. For example, if you run the command below from the `ubuntu-14.04-cpu-all-options` folder you will get a docker image around 1.5GB that has many optional libraries like OpenCV, for the minimal install, `ubuntu-14.04-cpu-minimal`, it is about 1GB and is just enough to run Caffe2, and finally for the gpu dockerfile, `ubuntu-14.04-gpu-all-options`, it is based on the NVIDIA CUDA docker image about 3.2GB and contains all of the optional dependencies. In a terminal window in one of those folders, simply run the following:

```
docker build -t caffe2 .
```

Don't miss the `.` as it is pointing to the `Dockerfile` in your current directory. Also, you can name docker image whatever you want. The `-t` denotes tag followed by the name you want it called, in this case `caffe2`. If the build completed you should see this output:

```
Step 8/8 : RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
 ---> Running in 0ca0a35635b8
Success
 ---> 5ee1fb669aef
Removing intermediate container 0ca0a35635b8
Successfully built 5ee1fb669aef
```

If you see "Success" just after the following test, then Caffe2 is working correctly.

```bash
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

Don't worry about the `Running in 0ca0a35635b8` as that is a temporary container specific to your build process. Also, the step numbers will vary depending on the kind of build you chose.
Once the build process is complete you can run it by its name or by the last unique ID that was provided upon completion. In this example case, this ID is `5ee1fb669aef`. To run the image in a container and get to bash you can launch it interactively using the following:

```bash
docker run -it caffe2 /bin/bash
```

You can run specific Caffe2 commands as well by hitting the Python interface directly or by interacting with IPython.

```bash
docker run -it caffe2 ipython
```

Then once in the IPython environment you can interact with Caffe2.

```python
In [1]: from caffe2.python import workspace
```

You may also try fetching some models directly and running them as described in this [Tutorial](../tutorials/Loading_Pretrained_Models.ipynb).

If you decide to try out the different Docker versions of Caffe2 using different dependencies then you will want to build them with their own tag and launch them using their tag or unique ID instead, for example:

```bash
docker run -it 5ee1fb669aef /bin/bash
```

**Docker - Ubuntu 14.04 with full dependencies notes:**

    - librocksdb-dev not found. (May have to install this yourself if you want it.)
