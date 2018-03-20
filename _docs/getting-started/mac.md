{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac prebuilt" />

We only support Anaconda packages at the moment. If you do not wish to use Anaconda, then you must build Caffe2 from [source](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile).

### Anaconda packages

We build Mac packages without CUDA support for both Python 2.7 and Python 3.6. To install Caffe2 with Anaconda, simply activate your desired conda environment and run the following command.

```bash
conda install -c caffe2 caffe2
```

> This does NOT include libraries that are necessary to run the tutorials, such as jupyter. See the [tutorials](https://caffe2.ai/docs/tutorials) page for the list of required packages needed to run the tutorials.

NOTE: This will install Caffe2 and all of its required dependencies into the current conda environment. We strongly suggest that you create a new conda environment and install Caffe2 into that. A conda environment is like its own python installation that won't have library version problems with your other conda environments. You can learn more about conda environments [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

To see what libraries these Caffe2 conda packages are built against, see `conda/no_cuda/meta.yaml`. If you want to use different libraries then you must build from  source.

### Prebuilt Caffe2 Python Wheel

No wheel is available at this time.


<block class="mac compile" />

There are three ways to install on Mac, with [Anaconda and conda](#anaconda-install-path), with [Anaconda but custom make commands](#custom-anaconda-install), or [without Anaconda](#brew-and-pip-install-path). Always pull the latest from GitHub, so you get any build fixes. See the [Troubleshooting](#troubleshooting) section below for tips.

## Anaconda Install Path

[Anaconda](https://www.continuum.io/downloads) is the recommended install route.  The following commands will install Caffe2 wherever your other conda packages are installed.

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
CONDA_INSTALL_LOCALLY=1 ./scripts/build_anaconda.sh
```

This will build Caffe2 using [conda build](https://conda.io/docs/user-guide/tasks/build-packages/recipe.html), with the flags specified in `conda/no_cuda/build.sh` and the packages specified in `conda/no_cuda/meta.yaml`. To build Caffe2 with different settings, change the dependencies in `meta.yaml` and the `CMAKE_ARGS` flags in `conda/no_cuda/build.sh` and run script again. Note that this will create a new ephemeral conda environment on every build, so it'll be slower than the [Custom Anaconda Installation](#custom-anaconda-install) approach below.

### Optional GPU Support

In the instance that you have a NVIDIA supported GPU in your Mac, then you should visit the NVIDIA website for [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and install the provided binaries. If you want to use `conda-build` then use `CONDA_INSTALL_LOCALLY=1 BUILD_ENVIRONMENT=cuda ./scripts/build_anaconda.sh` instead.

## Custom Anaconda Install

If you plan to change the source code of Caffe2 frequently and don't want to wait for a full conda build and install cycle, you may want to bypass conda and call Cmake manually. The following commands will build Caffe2 in a directory called `build` under your Caffe2 root and install Caffe2 in a conda env. **In this example Anaconda is installed in `~/anaconda2`,** if your Anaconda has a different root directory then change that in the code below.

```bash
# Create a conda environment
conda create -n my_caffe2_env && source activate my_caffe2_env

# Install required packages
# mkl isn't actually required, but is really recommended for Anaconda
conda install -y \
    future \
    gflags \
    glog \
    lmdb \
    mkl \
    mkl-include \
    numpy \
    opencv \
    protobuf \
    snappy \
    six

# Clone Caffe2
cd ~ && git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2

# Make Caffe2 in a separate folder to avoid polluting the Caffe2 source tree.
# This can be anywhere you want it
rm -rf build && mkdir build && cd build

# Add flags to this command to control which packages you want to use. The
# options to use are at the top of CMakeLists.txt in the topmost Caffe2
# directory
cmake -DUSE_CUDA=OFF -DUSE_LEVELDB=OFF -DCMAKE_PREFIX_PATH=~/anaconda2/envs/my_caffe2_env -DCMAKE_INSTALL_PREFIX=~/anaconda2/envs/my_caffe2_env ..
make install
```

The flag `CMAKE_PREFIX_PATH` tells Cmake to look for packages in your conda environment before looking in system install locations (like `/usr/local`); you almost certainly want to set this flag, since `conda install` installs into the activated conda environment. `CMAKE_INSTALL_PREFIX` tells Cmake where to install Caffe2 binaries such as `libcaffe2.dylib` after Caffe2 has been successfully built; the default is `/usr/local` (which probably isn't what you want).

If you do this, know that Cmake will cache things in this build folder, so you may want to remove it before rebuilding.

## Brew and Pip Install Path

Follow these instructions if you want to build Caffe2 without Anaconda. Make sure when you use pip that you're pointing to a specific version of Python or that you're using environments. You can check which version of pip that you are using with `which pip` and `pip --version`.

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
    numpy \
    protobuf \
    pydot \
    python-gflags \
    pyyaml \
    scikit-image \
    setuptools \
    six
```

### Clone and Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
```

We're going to build without CUDA, using the `-DUSE_CUDA=OFF` flag, since it would be rare at this point for your Mac to have GPU card with CUDA support.

```bash
# This will build Caffe2 in an isolated directory so that Caffe2 source is
# unaffected
mkdir build && cd build

# This configures the build and finds which libraries it will include in the
# Caffe2 installation. The output of this command is very helpful in debugging
cmake -DUSE_CUDA=OFF ..

# This actually builds and installs Caffe2 from makefiles generated from the
# above configuration step
sudo make install
```

## Test the Caffe2 Installation
Run this to see if your Caffe2 installation was successful. 

```bash
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

If this fails, then get a better error message by running a Python interpreter in the `caffe2/build` directory and then trying `from caffe2.python import core`.

## Troubleshooting

### Python Configuration

You might need to setup a PYTHONPATH environment variable. `echo $PYTHONPATH` and if your install prefix (default is `/usr/local`) is not there add `export PYTHONPATH=/usr/local` to `.zshrc`, `.bash_profile` or whatever you're using. If you built Caffe2 with CMAKE_INSTALL_PREFIX set to something else, then use that instead of '/usr/local'.

### Protobuf errors
Protobuf version mismatch is a common problem. Having different protobuf
versions often leads to incompatible headers and libraries.

Run these commands to see which protobuf is your default (if you are using conda environments, then the current conda environment affects the output of these commands).

```bash
which protoc
protoc --version
```

Run these commands to find other protobuf installations that may be causing problems.

```bash
find /usr -name libprotobuf* 2>/dev/null
find ~ -name libprotobuf* 2>/dev/null
```
Brew installs protobuf into `/usr/local` by default. Anaconda installs protobuf somewhere under the `anaconda` root folder (the command above assumes that you installed Anaconda into your home directory, as is recommended by Anaconda). 

If you want to use the protobuf in your conda environment but the installation keeps picking up a different protobuf, make sure your are calling cmake with CMAKE_PREFIX_PATH pointing to your conda environment and delete your build folder to delete the cmake cache.

If you can't figure out what's wrong, then the easiest way to fix protobuf problems is to uninstall all protobuf versions and then reinstall the one that you want to use. For example, if you want to use the protobuf in Anaconda's conda-forge, you could try

```bash
brew uninstall protobuf
pip uninstall protobuf
conda uninstall -y protobuf
conda install -y -c conda-forge protobuf
```

The trickiest part is during the linking of Caffe2. Once Caffe2 is built and installed, libcaffe2.dylib should point to the protobuf installed by conda (via a @rpath relative to @loader_path), and so should be unaffected by other protobuf versions on your machine. If it's not possible to permanently uninstall other protobuf versions, try temporarily uninstalling other protobuf versions while you make Caffe2, and then reinstall them afterwards.

### General debugging tips

Find things with `find`. On Mac's the conventional name of a library for a package `mypackage` is `libmypackage.a` or `libmypackage.dylib`. `find` accepts wildcards `*` that match any string of any length. For example

```bash
# Find everything associated with protobuf anywhere
find / -name *protobuf* 2>/dev/null

# Find all protobuf libraries everywhere
find / -name libprotobuf* 2>/dev/null
```

Use `which` in combination with the `--version` flag to find out more about executables on your system. For example

```bash
which python
python --version

which protoc
protoc --version
```

Use `otool -l` on libraries (usually .a or .dylib) to find out what other libraries it needs and where it expects to find them. Libraries are usually installed under `/usr/lib`, or `/usr/local/lib` (for Homebrew), or in various places under your anaconda root directory. You can find where libraries are with the `find` command above. otool example:

```bash
otool -l <path to libcaffe2.dylib>
```

This command can output a lot. To diagnose dynamic linking issues, look for `LC_LOAD_DYLIB` commands, they should look something like:

```
          cmd LC_LOAD_DYLIB
      cmdsize 80
         name /usr/local/opt/protobuf@3.1/lib/libprotobuf.11.dylib (offset 24)
   time stamp 2 Wed Dec 31 16:00:02 1969
      current version 12.0.0
compatibility version 12.0.0
Load command 11
```

In the example above, this library will look for protobuf in `/usr/local/opt` when it is loaded. In the example below, it will look for `libprotobuf` relative to the `@rpath`, which is set to `@loader_path` (see [dyld](https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man1/dyld.1.html)).

```
          cmd LC_LOAD_DYLIB
      cmdsize 56
         name @rpath/libprotobuf.14.dylib (offset 24)
   time stamp 2 Wed Dec 31 16:00:02 1969
      current version 15.0.0
compatibility version 15.0.0
Load command 11

... <output omitted> ...

          cmd LC_RPATH
      cmdsize 32
         path @loader_path/ (offset 12)
```


### Common errors

|Python errors
----|-----
Python environment | You may have another version of Python installed or need to support Python version 3 for other projects.
Solution | Try virtualenv or Anaconda. The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.
pip version | If you plan to use Python with Caffe2 then you need pip or Anaconda to install packages.
Solution | `pip` comes along with [Homebrew's python package](https://brew.sh/) or `conda` with [Anaconda](https://www.continuum.io/downloads).

|Building from source
----|-----
Homebrew | Test that your terminal is ready with `brew install wget`.
Solution | Run this to install Homebrew: `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
OS version | Caffe2 is known to work on Sierra (others TBD after testing)
git | While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.
Solution | `brew install git`
xcode | You may need to install [Xcode](https://developer.apple.com/xcode/) or at a minimum xcode command line tools.
Solution | You can install it via terminal using `xcode-select --install`
NNPACK | You may experience errors related to confu or PeachPy when running `make install`.
Solution | Install dependencies of NNPACK: `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/PeachPy` and `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/confu`

|GPU Support
----|-----
GPU | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for MacOS X.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone): install from NVIDIA's site; free developer account required. [NVIDIA MacOS X Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)

|Misc.
----|-----
malloc error | If you are using homebrew leveldb on a Mac, you might see an error warning you that malloc_zone_unregister() failed. This is not a Caffe2 issue but is due to the homebrew leveldb having an incompatible memory allocator. It does not affect usage.

{{ outro | markdownify }}

<block class="mac docker" />

<block class="mac cloud" />
