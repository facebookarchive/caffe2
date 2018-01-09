{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="mac compile" />

The Mac build works easiest with Anaconda. Always pull the latest from github, so you get any build fixes. See the Troubleshooting section below for tips.

## Anaconda Install Path

[Anaconda](https://www.continuum.io/downloads). Python 2.7 version is needed for Caffe2, and Anaconda is recommended. Skip this section to find brew/pip install directions if you are not using Anaconda.

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
conda build conda
conda install caffe2 --use-local
```

This will build caffe2 using [conda build](https://conda.io/docs/user-guide/tasks/build-packages/recipe.html), with the flags specified in `conda/build.sh` and the packages specified in `conda/meta.yaml`. `conda build` will create a conda package (tarball) on your machine, which `conda install` then installs. To build caffe2 with different settings, change the dependencies in `meta.yaml` and the `CMAKE_ARGS` flags in `conda/build.sh` and run the above command again. 

If your default Anaconda Python is not 2.7, you can install a different version of Python using `conda create --name python2 python=2` (`python2` can be any name you like.)  Subsequently, if you `source activate python2`, your path will be adjusted so that you get `python2`. See [this page](https://conda.io/docs/user-guide/tasks/manage-environments.html) on managing conda environments for more info.

### Optional GPU Support

In the instance that you have a NVIDIA supported GPU in your Mac, then you should visit the NVIDIA website for [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and install the provided binaries. You will have to remove `-DUSE_CUDA=OFF` and `-DUSE_NCCL=OFF` flags from `conda/build.sh`.

### Optional Dependencies

The following optional dependencies can be installed with the following command, or by adding the libraries to `conda/meta.yaml`.

```bash
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

## Brew and Pip Install Path

Follow these instructions if you want to build Caffe2 without Anaconda. Make sure when you use pip that you're pointing to a specific version of Python or that you're using environments. You can check which version of pip that you are using with `which pip` and `pip --version`.

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
future \
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
six \
tornado
```

### Clone and Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
```

We're going to build without CUDA, using the `-DUSE_CUDA=OFF` flag, since it would be rare at this point for your Mac to have GPU card with CUDA support.

```
mkdir build && cd build
cmake -DUSE_CUDA=OFF ..
sudo make install
```

## Test the Caffe2 Installation
Now test Caffe2 by running (in the `caffe2/build` directory)

```
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

If this fails, then get a better error message by running a Python interpreter in the `caffe2/build` directory and then trying `from caffe2.python import core`.

## Troubleshooting

### Python Configuration

You might need setup a PYTHONPATH environment variable. `echo $PYTHONPATH` and if it's not in there add `export PYTHONPATH=/usr/local` to `.zshrc`, `.bash_profile` or whatever you're using.

Change to a different folder and test Caffe2 again. If you are using Anaconda or had multiple versions of Python on your system the test may fail once out of the build folder. You will want to update the Python bindings:

```
sudo install_name_tool -change libpython2.7.dylib ~/anaconda/lib/libpython2.7.dylib /usr/local/caffe2/python/caffe2_pybind11_state.so
```

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

The easiest way to fix protobuf problems is to uninstall all protobuf versions and then reinstall the one that you want to use. For example, if you want to use the protobuf in Anaconda's conda-forge, you could try

```bash
brew uninstall protobuf
pip uninstall protobuf
conda uninstall -y protobuf
conda install -y -c conda-forge protobuf
```

The trickiest part is during the linking of caffe2. Once caffe2 is built and installed, libcaffe2.dylib should point to the protobuf installed by conda (via a @rpath relative to @loader_path), and so should be unaffected by other protobuf versions on your machine. If it's not possible to permanently uninstall other protobuf versions, try temporarily uninstalling other protobuf versions while you make caffe2, and then reinstall them afterwards.

### General debugging tips

Find things with `find`. On Mac's the conventional name of a library for a package `mypackage` is `libmypackage.a` or `libmypackage.dylib`. `find` accepts wildcards `*` are wildcards that match any string of any length. For example

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
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require [Python2.7](https://www.python.org/download/releases/2.7/). macOS Sierra comes pre-installed with Python 2.7.10, but you may need to update to run Caffe2. To check your version: `python --version`
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
NNPACK | You may experience errors related to confu or PeachPy when running `make install`.
Solution | Install dependencies of NNPACK: `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/PeachPy` and `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/confu`

|GPU Support
----|-----
GPU | The easiest route is to go to [NVIDIA's site and download](https://developer.nvidia.com/cuda-downloads) and install their binary for MacOS X.
Solution | Caffe2's GPU support is [Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone): install from NVIDIA's site; free developer account required. [NVIDIA MacOS X Installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/)

|Misc.
----|-----
malloc error | If you are using homebrew leveldb on a Mac, you might see an error warning you that malloc_zone_unregister() failed. This is not a caffe2 issue but is due to the homebrew leveldb having an incompatible memory allocator. It does not affect usage.

{{ outro | markdownify }}

<block class="mac prebuilt" />

### Prebuilt Caffe2 Python Wheel

No wheel is available at this time.


<block class="mac docker" />

<block class="mac cloud" />
