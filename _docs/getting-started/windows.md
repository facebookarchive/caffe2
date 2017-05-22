<block class="windows compile prebuilt" />

Windows 10 or greater is required to run Caffe2.

<block class="windows prebuilt" />

## Prebuilt

No binaries available.

<block class="windows compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

Windows build is in testing and beta mode. For the easiest route, use the docker images for now in CPU-only mode.

## Required Dependencies

The first thing you want to do is to assess whether or not you're going to use GPU acceleration with Caffe2. If you have an [NVIDIA GPU](https://www.nvidia.com/en-us/deep-learning-ai/solutions/) and you plan on training some neural networks, then it's probably worth the extra installation an effort. If you're just going to play around with pre-trained models then skip the video drivers and NVIDIA CUDA/cuDNN installation steps.

1. **Update your video drivers**: assuming you have an NVIDIA card, use NVIDIA GeForce Experience to run the latest update.
2. [NVIDIA CUDA/cuDNN](https://developer.nvidia.com/cuda-downloads): if you have GPU(s) then go ahead and install
    * [CUDA](https://developer.nvidia.com/cuda-downloads)
    * [cuDNN](https://developer.nvidia.com/cudnn) (registration required; it is a zip file, not installer, so you need to copy the contents of the zip file to the cuda folder which is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0` by default)
3. Python 2.7.6 to [Python 2.7.13](https://www.python.org/download/releases/python-2713/). Python version 3+ is not yet supported. You can use regular Python or Anaconda Python. Just note that you many have issues with package location and versioning with Anaconda. Some Anaconda notes are provided below the Regular Python notes.
4. Install a C++ compiler such as [Visual Studio Community Edition 2017](https://www.visualstudio.com/vs/community/)
    * When installing VS 2017, install Desktop Development with C++ (on the right check: C++/CLI support)
5. Install [Cmake](http://cmake.org)
6. Run `Developer Command Prompt for VS 2017`.
7. Install `protobuf`. Go to `caffe2\scripts\` and run `build_host_protoc.bat`. This should build protobuf from source for your system.

## Setup Python, Install Python Packages, Build

### Regular Python Install

Install [Python 2.7.13](https://www.python.org/download/releases/python-2713/) and [Microsoft Visual C++ Compiler for Python 2.7](http://aka.ms/vcpython27).

Assuming you have already added `C:\Python27` and `C:\Python27\scripts` to your Path environment variable, you can go ahead and use pip to install the Python dependencies.

```
pip install numpy ^
            protobuf ^
            hypothesis
```

While these are optional, they're recommended if you want to run the tutorials and utilize much of the provided materials.

```
pip install flask ^
            glog ^
            graphviz ^
            jupyter ^
            matplotlib ^
            pydot python-nvd3 ^
            pyyaml ^
            requests ^
            scikit-image ^
            scipy ^
            setuptools ^
            tornado
```

** Unresolved Issues with Optional Packages **

* gflags: need to build from source
* glog: need to build from source
* leveldb: need to build from source

**leveldb build notes:**

* Download Boost and build it, specifying static libraries (the default is shared) and 64 bit if necessary (32 bit is default)
* Get the qdb branch of leveldb: https://github.com/bureau14/leveldb
* Build leveldb, ensuring Runtime Library is set to 'Multi-Threaded (/MT)' in properties,C/C++ for both the leveldb and leveldbutil projects
* Download the Windows port of Snappy for C++

### Clone & Build

Open up a Developer Command Prompt, find an appropriate place to clone the repo, and use this command. Or if you have github desktop, you can use that instead. If you've already forked Caffe2 or have it locally and you're using Visual Studio, skip ahead to the next step.

```
git clone --recursive https://github.com/caffe2/caffe2.git
```

Using the Developer Command Prompt, browse to the repo's folders to `\caffe2\scripts` and run `build_windows.bat`.

For VS15 and VS17 users:

1. Install the [Github Extension for Visual Studio](https://visualstudio.github.com).
2. From within Visual Studio you can open/clone the Github repository. From the Getting Started page under Open, you should have GitHub as an option. Login, and then either choose Caffe2 from the list (if you've forked it) or browse to where you cloned it. Default location is hereinafter is referencing `C:\Users\username\Source\Repos\caffe2`.

### Python Configuration

You will find the Caffe2 binary in `$USER\Source\Repos` (if that's where you put the caffe2 source) `\caffe2\build\caffe2\binaries\Release`

To get python to recognize the DLL, rename `caffe2_pybind11_state.dll` from `.dll` to `.pyb` and copy it to Python's DLL folder `$USER\AppData\Local\Continuum\Anaconda2\DLLs`. If you're not using Anaconda, then look for it in your Python27 or python-2713 folder.


### Anaconda Python

** this install path needs correction / confirmation **

1. [Anaconda](https://www.continuum.io/downloads): download the Python 2.7 version.
2. Run `Anaconda Prompt` as Administrator. Go to the search bar, search for "anaconda prompt" and right-click it and choose "Run as Administrator".
3. Install Python packages:

```
conda install -y --channel https://conda.anaconda.org/conda-forge  graphviz hypothesis numpy pydot-ng python-lmdb requests zeromq
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
