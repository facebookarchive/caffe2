{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac prebuilt" />

We only support Anaconda packages at the moment. If you do not wish to use Anaconda, then you must build Caffe2 from [source](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile).

### Anaconda packages

We build Mac packages without CUDA support for both Python 2.7 and Python 3.6. To install Caffe2 with Anaconda, simply activate your desired conda environment and run the following command.

```bash
conda install -c caffe2 caffe2
```

> This does NOT include libraries that are necessary to run the tutorials, such as jupyter. See the [tutorials](https://caffe2.ai/docs/tutorials) page for the list of required packages needed to run the tutorials.

NOTE: This will install Caffe2 and all of its required dependencies into the current conda environment. We strongly suggest that you create a new conda environment and install Caffe2 into that. A conda environment is like a separate python installation and so won't have problems with your other conda environments. You can learn more about conda environments [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

To see what libraries these Caffe2 conda packages are built against, see `conda/no_cuda/meta.yaml`. If you want to use different libraries then you must build from  source.

### Prebuilt Caffe2 Python Wheel

No wheel is available at this time.


<block class="mac compile" />

To compile Caffe2 to use your GPU, follow [these](#gpu-support) instructions first, then:

* Follow [these](#anaconda-install-path) instructions if you have Anaconda.
* Follow [these](#brew-and-pip-install-path) instructions if you do not have Anaconda.
* (Advanced) Follow [these](#custom-anaconda-install) instructions if you have Anaconda, are used to working with Anaconda environments, are familiar with how Anaconda installs and finds dependencies, and need to change Caffe2 c++ source and recompile frequently.

For any problems, see our [troubleshooting guide](faq.html).

## Anaconda Install Path

[Anaconda](https://www.continuum.io/downloads) is the recommended install route.  The following commands will install Caffe2 wherever your other conda packages are installed.

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
CONDA_INSTALL_LOCALLY=1 ./scripts/build_anaconda.sh
```

This will build Caffe2 using [conda build](https://conda.io/docs/user-guide/tasks/build-packages/recipe.html), with the flags specified in `conda/no_cuda/build.sh` and the packages specified in `conda/no_cuda/meta.yaml`. To build Caffe2 with different settings, change the dependencies in `meta.yaml` and the `CMAKE_ARGS` flags in `conda/no_cuda/build.sh` and run the script again.

If you want to build with GPU, then use `CONDA_INSTALL_LOCALLY=1 BUILD_ENVIRONMENT=conda-cuda-macos ./scripts/build_anaconda.sh` instead.

Now [test your Caffe2 installation](#test-the-caffe2-installation).


## Brew and Pip Install Path

> If Anaconda is installed on your system, then please use the [Anaconda install route](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile#anaconda-install-path). Otherwise it is easy to encounter version mismatch issues.

### Install Caffe2's dependencies

You will need [brew](https://brew.sh/) to install Caffe2's dependencies.

```bash
brew install \
    automake \
    cmake \
    git \
    gflags \
    glog \
    python
```

```bash
sudo -H pip install \
    future \
    leveldb \
    numpy \
    protobuf \
    pydot \
    python-gflags \
    pyyaml \
    scikit-image \
    six
```

> To run the tutorials you will need to install more packages, see the [Tutorial](https://caffe2.ai/docs/tutorials) page for full requirements.

### Clone and Build

```bash
# Clone Caffe2's source code from our Github repository
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2

# Create a directory to put Caffe2's build files in
mkdir build && cd build

# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake ..

# Compile, link, and install Caffe2
sudo make install
```

Now [test your Caffe2 installation](#test-the-caffe2-installation).

## Custom Anaconda Install

> Only use this path if you have Anaconda and plan to change Caffe2 source and rebuild frequently. It is easiest to encounter version mismatch or incompatibility issues using this approach.

If you plan to change the source code of Caffe2 frequently and don't want to wait for a full conda build and install cycle, you may want to bypass conda and call Cmake manually. The following commands will build Caffe2 in a directory called `build` under your Caffe2 root and install Caffe2 in a conda env. **In this example Anaconda is installed in `~/anaconda2`,** if your Anaconda has a different root directory then change that in the code below.

```bash
# Create a conda environment
conda create -yn my_caffe2_env && source activate my_caffe2_env

# Install required packages
conda install -y \
    future \
    gflags \
    glog \
    leveldb \
    mkl \
    mkl-include \
    numpy \
    opencv \
    protobuf \
    six

# Clone Caffe2's source code from our Github repository
cd ~ && git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2

# Create a directory to put Caffe2's build files in
rm -rf build && mkdir build && cd build

# Configure Caffe2's build
# This looks for packages on your machine and figures out which functionality
# to include in the Caffe2 installation. The output of this command is very
# useful in debugging.
cmake -DCMAKE_PREFIX_PATH=~/anaconda2/envs/my_caffe2_env -DCMAKE_INSTALL_PREFIX=~/anaconda2/envs/my_caffe2_env ..

# Compile, link, and install Caffe2
make install
```

The flag `CMAKE_PREFIX_PATH` tells Cmake to look for packages in your conda environment before looking in system install locations (like `/usr/local`); you almost certainly want to set this flag, since `conda install` installs into the activated conda environment. `CMAKE_INSTALL_PREFIX` tells Cmake where to install Caffe2 binaries such as `libcaffe2.dylib` after Caffe2 has been successfully built; the default is `/usr/local` (which probably isn't what you want).

If you do this, know that Cmake will cache things in this build folder, so you may want to remove it before rebuilding.

## Test the Caffe2 Installation
Run this to see if your Caffe2 installation was successful. 

```bash
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

If this fails, then get a better error message by running a Python interpreter in the `caffe2/build` directory and then trying `from caffe2.python import core`. Then see the [Troubleshooting](faq.html) page for help.

## GPU Support

In the instance that you have a NVIDIA supported GPU in your Mac, then you should visit the NVIDIA website for [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and install the provided binaries. Also see this [Nvidia guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/) on setting up your GPU correctly. Caffe2 requires CUDA 6.5 or greater.

Once CUDA and CuDNN (and optionally NCCL) are installed, please verify that your CUDA installation is working as expected, and then continue with your preferred Caffe2 installation path. 

After Caffe2 is installed, you should NOT see the following error when you try to import caffe2.python.core in Python

```bash
WARNING:root:This caffe2 python run does not have GPU support. Will run in CPU only mode.
WARNING:root:Debug message: No module named 'caffe2.python.caffe2_pybind11_state_gpu'
```

If you see this error then your GPU installation did not work correctly.

{{ outro | markdownify }}

<block class="mac docker" />

<block class="mac cloud" />
