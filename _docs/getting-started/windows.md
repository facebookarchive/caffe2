<block class="windows compile prebuilt" />

Windows 10 or greater is required to run Caffe2.

<block class="windows prebuilt" />

## Prebuilt

** LINK TO WINDOWS BINARY HERE **

<block class="windows compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

### Required Dependencies

The first thing you want to do is to assess whether or not you're going to use GPU acceleration with Caffe2. If you have an NVIDIA GPU and you plan on training some neural networks, then it's probably worth the extra installation an effort. If you're just going to play around with pre-trained models then skip the video drivers and CUDA/cuDNN installation steps.

1. **Update your video drivers**: assuming you have an NVIDIA card, use NVIDIA GeForce Experience to run the latest update.
2. [NVIDIA CUDA/cuDNN](https://developer.nvidia.com/cuda-downloads): if you have GPU(s) then go ahead and install
    * [CUDA](https://developer.nvidia.com/cuda-downloads)
    * [cuDNN](https://developer.nvidia.com/cudnn) (registration required; zip file, not installer)
3. Python: you have a couple of options, plain Python or Anaconda, but we'll recommend Anaconda
    * [Anaconda](https://www.continuum.io/downloads)
    * [Python 2.7.13](https://www.python.org/download/releases/python-2713/): this includes `pip` which is used to install many other dependencies. Most of the time you can swap conda for pip in the directions below.
4. Install a C++ compiler such as [Visual Studio Community Edition 2017](https://www.visualstudio.com/vs/community/)
    * When installing VS 2017, install Desktop Development with C++ (on the right check: C++/CLI support)
5. Install [Cmake](http://cmake.org)
6. Install Python packages numpy and protobuf. From a command line:

```
conda install -y numpy
conda install -y --channel https://conda.anaconda.org/eugene protobuf
```

### Optional Dependencies

While these are optional, they're recommended if you want to run the tutorials and utilize much of the provided materials.

**Python optional dependencies:**

#### Install glog, gflags and other related dependencies

```
conda install -y --channel https://conda.anaconda.org/willyd glog
```

#### Install leveldb

Note that this upgrades protobuf to 3.1 which needs to be downgraded to 2.5 (not sure if this breaks things)

```
conda install -y --channel https://conda.anaconda.org/willyd leveldb
```

#### Install other options

```
conda install -y --channel https://conda.anaconda.org/conda-forge  graphviz hypothesis pydot-ng python-lmdb zeromq
```

#### Overlay protobuf v2.5.0 if you installed something that upgraded it

```
conda install -y --channel https://conda.anaconda.org/eugene protobuf
```

#### Things that are options that get installed already by other prerequisites

  * flask (already installed)
  * matplotlib (probably can skip as it comes with numpy)
  * pyyaml (already installed)
  * scikit-image (probably can skip - comes with numpy)
  * scipy (probably can skip - comes with numpy)
  * setuptools (already installed)
  * tornado (already installed)

#### Not found for Windows with conda - not the end of the world, but if you want these you'll probably have to build them from source.

  * automake
  * open-mpi
  * python-nvd3
  * rocksdb

The pip route if you want to try that instead. Note that there's no a 1:1 on these options between pip and conda:

```
python -m pip install flask graphviz hypothesis jupyter matplotlib pydot python-nvd3 pyyaml scikit-image scipy setuptools tornado
```

### Clone & Build

Open up a command prompt, find an appropriate place to clone the repo, and use this command. Or if you have github desktop, you can use that instead. If you've already forked Caffe2 or have it locally and you're using Visual Studio, skip ahead to the next step.

```
git clone --recursive https://github.com/caffe2/caffe2.git
```

For VS15 and VS17 users:

1. Install the [Github Extension for Visual Studio](https://visualstudio.github.com).
2. From within Visual Studio you can open/clone the github repository. From the Getting Started page under Open, you should have GitHub as an option. Login, and then either choose Caffe2 from the list (if you've forked it) or browse to where you cloned it. Default location is hereinafter is referencing `C:\Users\username\Source\Repos\caffe2`.

**Then run this batch file to start the build.**

### Python Configuration

You will find the Caffe2 binary in `$USER\Source\Repos` (if that's where you put the caffe2 source) `\caffe2\build\caffe2\binaries\Release`

To get python to recognize the DLL, rename `caffe2_pybind11_state.dll` from `.dll` to `.pyb` and copy it to Python's DLL folder `$USER\AppData\Local\Continuum\Anaconda2\DLLs`. If you're not using Anaconda, then look for it in your Python27 or python-2713 folder.


### Troubleshooting

| Build errors
----|-----
C++ compiler not found | For VS 2017 users, update the windows install batch file for the -G switch found in (caffe2/scripts/build_windows.bat).
Solution | Note the cmake section and update it to reflect your VS version: `cmake -G “Visual Studio 15 2017 Win64”`


| Python errors
----|-----
'python' is not recognized... | You need to setup Python in your PATH environment variable.
Solution | Depending on you version of Windows, you go about this differently. Generally this is Control Panel > System & Security > System > Advanced system settings > Environment Variables, edit the PATH variable and add a new entry for `C:\Python27` or whatever you installation directory was for Python. You're looking for wherever python.exe resides.


| GPU Support
----|-----
GPU drivers | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for Windows.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone). CUDA 8.0 is recommended. Install from NVIDIA's site; free developer account required. [NVIDIA Windows Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
Installing CUDA 8.0: “No supported version of Visual Studio was found. Must use VS 15.” | Until NVIDIA updates CUDA to support VS 17, you're going to have to install VS 15 and try again.


<block class="windows docker" />

### Dependencies

* If you have Windows 10 Professional, then install [Docker Community Edition for Windows](http://store.docker.com)
* If you have a Windows 10 Home, then you need [Docker Toolbox](https://www.docker.com/products/docker-toolbox)

**Note: GPU mode is [not currently supported](https://github.com/NVIDIA/nvidia-docker/issues/197) with Docker on Windows with the possible exception of Windows Server 2016**

<block class="windows cloud" />
