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
3. [Python 2.7.13](https://www.python.org/download/releases/python-2713/): this includes `pip` which is used to install many other dependencies.
4. Install a C++ compiler such as [Visual Studio Community Edition 2017](https://www.visualstudio.com/vs/community/)
5. Install Python packages numpy and protobuf. From a command line:

```
python -m pip install numpy protobuf
```

### Optional Dependencies

While these are optional, they're recommended if you want to run the tutorials and utilize much of the provided materials.

**Python optional dependencies:**

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


### Troubleshooting

|Python errors
----|-----
'python' is not recognized... | You need to setup Python in your PATH environment variable.
Solution | Depending on you version of Windows, you go about this differently. Generally this is Control Panel > System & Security > System > Advanced system settings > Environnltflrinekenunjftlinvirfblneureement Variables, edit the PATH variable and add a new entry for `C:\Python27` or whatever you installation directory was for Python. You're looking for wherever python.exe resides.

|Building from source
----|-----

| GPU Support
----|-----
GPU | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for Windows.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone). CUDA 8.0 is recommended. Install from NVIDIA's site; free developer account required. [NVIDIA Windows Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)



<block class="windows docker" />

### Dependencies

* If you have Windows 10 Professional, then install [Docker Community Edition for Windows](http://store.docker.com)
* If you have a Windows 10 Home, then you need [Docker Toolbox](https://www.docker.com/products/docker-toolbox)

**Note: GPU mode is [not currently supported](https://github.com/NVIDIA/nvidia-docker/issues/197) with Docker on Windows with the possible exception of Windows Server 2016**

<block class="windows cloud" />
