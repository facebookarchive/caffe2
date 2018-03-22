---
docid: faq
title: FAQ / Troubleshooting Help
layout: docs
permalink: /docs/faq.html
---

* [What operating systems are supported?](#what-operating-systems-are-supported)
* [What languages does Caffe2 support?](#what-languages-does-caffe2-support)
* [How do I use Caffe2 with my GPU?](#how-do-i-use-caffe2-with-my-gpu)
* [What are all of these optional libraries used for?](#what-are-all-of-these-optional-libraries-used-for)
* [Why do I get import errors in Python when I try to use Caffe2?](#why-do-i-get-import-errors-in-python-when-i-try-to-use-caffe2)
* [Why isn't Caffe2 working as expected in Anaconda?](#why-is-caffe2-not-working-as-expected-in-anaconda)
* [How do I fix error messages that are protobuf related?](#how-do-i-fix-error-messages-that-are-protobuf-related)
* [How can I find a file, library, or package on my computer?](#how-can-i-find-a-file-library-or-package-on-my-computer)
* [How can I find what dependencies my Caffe2 library (or other library) has?](#how-can-i-find-what-dependencies-my-caffe2-library-or-other-library-has)
* [The source directory does not contain a CMakeLists.txt file](#the-source-directory-does-not-contain-a-cmakeliststxt-file)
* [No module named caffe2_pybind11_state_gpu](#no-module-named-caffe2pybind11stategpu)
* [My python kernel keeps crashing when using Jupyter](#my-python-kernel-keeps-crashing-when-using-jupyter)
* [I still have a question, where can I get more help?](#i-still-have-a-question-where-can-i-get-more-help)

## What operating systems are supported?

Caffe2 is tested on

* macOS Sierra or above
* Ubuntu 14.04 and 16.04
* CentOS 7
* Windows 10
* iOS and Android
* Raspberry Pi
* Nvidia Tegra

## What languages does Caffe2 support?

Caffe2 is written in C++ with a Python frontend. You can find all of the code on our [Github page](https://github.com/caffe2/caffe2).

## How do I use Caffe2 with my GPU?

Many of Caffe2's operators have CUDA implementations, allowing you to use Caffe2 with your Nvidia GPU. To install Caffe2 with GPU support, first install all the needed Nvidia libraries ([CUDA](https://developer.nvidia.com/cuda-downloads) and CuDNN) and then follow the [installation instructions](https://caffe2.ai/docs/getting-started.html).

## What are all of these optional libraries used for?

Caffe2 can has many optional dependencies, which extend Caffe2's core functionality.

----|-----
[cuDNN](https://developer.nvidia.com/cudnn) | If using GPU, this is needed for Caffe2's cuDNN operators
[Eigen 3](http://eigen.tuxfamily.org/) | The default BLAS backend
[LevelDB](https://github.com/google/leveldb) | One of the DB options for storing Caffe2 models
[Nvidia CUDA](https://developer.nvidia.com/cuda-zone) | v6.5 or greater
[OpenCV](http://opencv.org/) | for image-related operations; requires leveldb <= v1.19
[OpenMPI](http://www.open-mpi.org/) | for MPI-related Caffe2 operators, which are used for distributed training
[RocksdB](http://rocksdb.org) | for Caffe2's RocksDB IO backend
[ZeroMQ](http://zeromq.org/) | needed for Caffe2's ZmqDB IO backend (serving data through a socket)
[Graphviz](http://www.graphviz.org/) | Used for plotting in the Jupyter Notebook Tutorials
[Hypothesis](https://hypothesis.readthedocs.io/) | Used in all of the tests
[Jupyter](https://ipython.org/) | Used for interactive python notebooks
[LevelDB](https://github.com/google/leveldb) | One of the DB options for storing Caffe2 models
[lmdb](https://lmdb.readthedocs.io/en/release/) | One of the DB options for storing Caffe2 models
[Matplotlib](http://matplotlib.org/) | Used for plotting in the Jupyter Notebook Tutorials
[Pydot](https://pypi.python.org/pypi/pydot) | Used for plotting in the Jupyter Notebook Tutorials
[Python-nvd3](https://pypi.python.org/pypi/python-nvd3/) | Used for plotting in the Jupyter Notebook Tutorials
[pyyaml](http://pyyaml.org/) | Used in the MNIST tutorial
[requests](http://docs.python-requests.org/en/master/) | Used in the MNIST tutorial to download the dataset
[Scikit-Image](http://scikit-image.org/) | Used for image processing
[SciPy](https://www.scipy.org/) | Used for assorted mathematical operations
[ZeroMQ](http://zeromq.org/) | needed for Caffe2's ZmqDB IO backend (serving data through a socket)


## Why do I get import errors in Python when I try to use Caffe2?

If you are trying to run a tutorial, then make sure you have installed all of the dependencies at the top of the [tutorials](tutorials.html) page. If you are getting an import error on `caffe2` itself, then you might have to set PYTHONPATH, but should understand the following first.

A Python installation consists of a Python executable (just called `python`) and a corresponding set of directories:

* **lib/** where c and c++ libraries are found
* **lib/python2.7/site-packages/** where installed Python modules are found (Python modules are just Python source files)
* **include/** where headers for c and c++ libraries are found
* **bin/** where the Python executable itself is found

Your python installation looks like this:

```
/usr/local/                                     # your python root and CMAKE_INSTALL_PREFIX
  +-- bin/
    +-- python
    +-- <other executables>
  +-- include/
    +-- <header files for c and c++ libraries>
  +-- lib/
    +-- libcaffe2.dylib
    +-- <other Caffe2 c and c++ libraries>
    +-- <other c and c++ libraries>
    +-- python2.7/                              # or python 3.6, or whatever your version is
      +-- site-packages/
        +-- caffe2/                             # where compiled Caffe2 python source files are
        +-- <other installed python modules>
```

Your python root is at "$(which python)/../.." . If you are using a conda environment then it will be `<Anaconda root directory>/envs/<conda env name>` instead of `/usr/local/` . If you are using Anaconda but not an environment, then it will be your Anaconda root directory instead of `/usr/local/`.

When Python imports anything, it first looks in its own site-packages directory and then looks in all directories in the PYTHONPATH environment variable. If you are having trouble importing Caffe2 in python (errors such as "ModuleNotFoundError"), then:

1. Make sure that you are using the same python that was used to build Caffe2. If you installed or uninstalled a new version of python or Anaconda since building Caffe2 then your python may have changed. If you are using Anaconda, make sure you are using the same conda environment that was used to build Caffe2.
2. If Caffe2 is installed into the correct site-packages (the directory structure looks like it does above) and python can still not import Caffe2, then add the current python root to PYTHONPATH by running `export PYTHONPATH="${PYTHONPATH}:$(which python)/../..` and try again.

If you overrode the default CMAKE_INSTALL_PREFIX when you built Caffe2, then you might have to add that to PYTHONPATH as well.


## Why is Caffe2 not working as expected in Anaconda?

Also answers **How do Anaconda Python installations work**

If you use Anaconda then your python installation is a little more complicated and will look more like

```
~/anaconda2/
  +-- bin/
    +-- python
  +-- include/
  +-- lib/
    +-- <c and c++ libraries installed in the root env>
    +-- python2.7/
      +-- site-packages/
        +-- <python modules installed in the root env>
  +-- envs/
    +-- my_caffe2_env/
      +-- bin/
        +-- python
      +-- include/
      +-- lib/
        +-- <c and c++ libraries installed in my_caffe2_env>
        +-- python2.7/
          +-- site-packages/
            +-- <python modules installed in my_caffe2_env>
    +-- <your other conda envs>
```

Whenever you make a new conda environment with `conda create <env_name>` then Anaconda adds a new directory `~/anaconda2/envs/<env_name>` with its own `bin/python`, `lib/` and `python2.7/site-packages/` . This is how Anaconda keeps each environment separate from each other.

Notice that there is still a complete Python installation in the root Anaconda directory. This is the "root" or "default" conda environment, and where `pip` and `conda` will install packages when you do not have conda environment activated. However, **conda envs will also look in the root env** if they do not find required modules in their own env. So if you install Protobuf into env my_caffe2_env and also into the root env and install Caffe2 into env my_caffe2_env and then uninstall Protobuf from my_caffe2_env, then Caffe2 will start using the Protobuf from your root env instead. This behaviour is often not what you want and can easily cause version mismatch problems. For this reason we recommend that you avoid installing packages into your root conda environment and install Caffe2 into a brand new conda environment.

## How do I fix error messages that are Protobuf related?

Protobuf version mismatch is a common problem. Having different protobuf versions often leads to incompatible headers and libraries.

Run these commands to see which protobuf is your default (if you are using conda environments, then the current conda environment affects the output of these commands).

```bash
which protoc
protoc --version
```

Run this commands to find other protobuf installations that may be causing problems.

```bash
find /usr -name libprotobuf.dylib 2>/dev/null
```

If you can't figure out what's wrong, then the easiest way to fix protobuf problems is to uninstall all protobuf versions and then reinstall the one that you want to use. For example, if you want to use the protobuf in Anaconda's conda-forge, you could try

```bash
brew uninstall protobuf
pip uninstall protobuf
conda uninstall -y protobuf
conda install -y -c conda-forge protobuf
```

## How can I find a file, library, or package on my computer?

Find libraries, binaries, and other files with the `find` command. On Macs and Linux, the conventional name of a library for a package `mypackage` is `libmypackage.a` or `libmypackage.dylib`. `find` accepts wildcards `*` that match any string of any length. For example

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

## How can I find what dependencies my Caffe2 library (or other library) has?

**For Linux, Ubuntu and CentOS**

Use `ldd <path to a library>` on libraries (files that end in .so) to find out what other libraries it needs and where it expects to find them. You can find where libraries are with the `find` command above.


**For macOS**

Use `otool -L <path to a library>` on libraries (files that end in .dylib) to find out what other libraries it needs and where it expects to find them. Libraries are usually installed under `/usr/lib`, or `/usr/local/lib` (for Homebrew), or alongside a [Python installation](#why-do-i-get-import-errors-in-python-when-i-try-to-use-caffe2). You can find where libraries are with the `find` command above. 

For example:

```bash
$ otool -L ~/anaconda3/envs/my_caffe2_env/lib/libcaffe2.dylib
/Users/my_username/anaconda3/envs/my_caffe2_env/lib/libcaffe2.dylib:
	@rpath/libcaffe2.dylib (compatibility version 0.0.0, current version 0.0.0)
	/usr/local/lib/libprotobuf.14.dylib (compatibility version 15.0.0, current version 15.0.0)
	@rpath/libmkl_rt.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libgflags.2.2.dylib (compatibility version 2.2.0, current version 2.2.1)
	/usr/local/lib/libglog.0.dylib (compatibility version 1.0.0, current version 1.0.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.0.0)
```

To find out what "@rpath" is, run otool with `-l` (lowercase L) and pipe that into `grep LC_RPATH -A 3`, e.g.

```bash
$ otool -l ~/anaconda3/envs/my_caffe2_env/lib/libcaffe2.dylib | grep LC_RPATH -A 3
          cmd LC_RPATH
      cmdsize 32
         path @loader_path/ (offset 12)
```

In this example, `@rpath` will evaluate to `@loader_path`, which is essentially the location of the library that you ran otool on (see [dyld](https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man1/dyld.1.html)). So, in this example, libcaffe2.dylib will look for libgflags.2.2.dylib in the same folder that libcaffe2.dylib is in.


## The source directory does not contain a CMakeLists.txt file

You need to run `git submodule update --init` in the Caffe2 root directory.


## No module named caffe2_pybind11_state_gpu

If you are not building for GPU then ignore this. If you are building for GPU, then make sure CUDA was found correctly in the output of the `cmake` command that was run to build Caffe2.


## My python kernel keeps crashing when using Jupyter

This happens when you try to call Jupyter server directly (like in a Docker container). Use `sh -c "jupyter notebook ..."` to get around this problem.


## I still have a question, where can I get more help?

For further issues, please post a new issue to our [issue tracker on Github](https://github.com/caffe2/caffe2/issues). 

> If your question is about an error installing Caffe2, then please include the following information in your issue:

* `$(uname -a)`
* Which installation guide you followed.
* What flags you passed to `cmake`
* The full output of your `cmake` command


## Miscellaneous errors

* On Mac, you may need to install [Xcode](https://developer.apple.com/xcode/) or at a minimum xcode command line tools. You can install it by running `xcode-select --install`
* If you experience errors related to confu or PeachPy when running `make install`, then install dependencies of NNPACK: `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/PeachPy` and `[sudo] pip install --upgrade git+https://github.com/Maratyszcza/confu`
* If you get model downloading errors when running in `sudo`, then PYTHONPATH might not be visible in that context. Run `sudo visudo` then add this line: `Defaults    env_keep += "PYTHONPATH"`
* If you encounter "AttributeError: 'module' object has no attribute 'MakeArgument'" when calling `core.CreateOperator` then try removing `caffe2/python/utils` from the directory that you installed Caffe2 in.
* If you see [errors with libdc1394](http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394) when opencv is installed, then run `ln /dev/null /dev/raw1394` . That solution is not [persistent](http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu) so try `sh -c 'ln -s /dev/null /dev/raw1394'` or when instantiating the container use: `--device /dev/null:/dev/raw1394`
