<block class="ubuntu compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2) 

### Required Dependencies

```bash
sudo apt-get install python-dev python-pip git build-essential cmake libprotobuf-dev protobuf-compiler libgoogle-glog-dev
sudo pip install numpy protobuf
```

### Optional GPU Support

If you plan to use GPU instead of CPU only, then you should install NVIDIA CUDA and cuDNN, a GPU-accelerated library of primitives for deep neural networks.
[NVIDIA's detailed instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) or if you're feeling lucky try the quick install set of commands below.

```bash
sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
sudo apt-get install cuda
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz" && curl -fsSL ${CUDNN_URL} -O
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```

### Optional Dependencies

```bash
sudo apt-get libgtest-dev libgflags2 libgflags-dev liblmdb-dev libleveldb-dev libsnappy-dev libopencv-dev libiomp-dev librocksdb-dev openmpi-bin openmpi-doc libopenmpi-dev
sudo pip install setuptools flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml
```

### Clone & Build

```
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

### Troubleshooting

|Python errors
----|-----
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require Python2.7. *Ubuntu 14.04 and greater have Python built in by default*, and that can be used to run Caffe2. To check your version: `python --version`
Solution | If you want the developer version of python, you could install the `dev` package for Python: `sudo apt-get install python-dev`
Python environment | You may have another version of Python installed or need to support Python version 3 for other projects.
Solution | Try virtualenv or Anaconda. The [Anaconda](https://www.continuum.io/downloads) platform provides a single script to install many of the necessary packages for Caffe2, including Python. Using Anaconda is outside the scope of these instructions, but if you are interested, it may work well for you.
pip version | If you plan to use Python with Caffe2 then you need pip.
Solution | `sudo apt-get install python-pip` and also try using pip2 instead of pip.

|Building from source
----|-----
OS version | Caffe2 requires Ubuntu 14.04 or greater.
git | While you can download the Caffe2 source code and submodules directly from GitHub as a zip, using git makes it much easier.
Solution | `sudo apt-get install git`
protobuf | You may experience an error related to protobuf during the make step.
Solution | Make sure you've installed protobuf in **both** of these two ways: `sudo apt-get install libprotobuf-dev protobuf-compiler && sudo pip install protobuf`

| GPU Support
----|-----
GPU errors | Unsupported GPU or wrong version
Solution | You need to know the specific `deb` for your version of Linux. `sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb` Refer to NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).
Build issues | Be warned that installing CUDA and cuDNN will increase the size of your build by about 4GB, so plan to have at least 12GB for your Ubuntu disk size.

| Caffe2 Python
----|-----
Module not found | Verify that Caffe2 was installed correctly
Solution | Run the following: `python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"` An output of `Success` means you are ready to with Caffe2 - congratulations! An output of `Failure` usually means you have not installed one of the dependencies correctly.
Dependencies missing | It's possible you're trying to run something that was using an optional dependency.
Solution | `sudo pip install setuptools flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml`
matplotlib error | Sometimes you need setuptools first: `sudo pip install -U pip setuptools && sudo pip install matplotlib`

| System Dependencies
----|-----
[Nvidia CUDA 6.5 or greater](https://developer.nvidia.com/cuda-zone) |
[C++ 11](https://en.wikipedia.org/wiki/C%2B%2B11) |
[Google Protocol Buffers](https://developers.google.com/protocol-buffers/) |
[Google Logging Module](https://github.com/google/glog) |
[gflags](https://gflags.github.io/gflags/) |
[Eigen 3](http://eigen.tuxfamily.org/) |
[NumPy](http://www.numpy.org/) |

| Optional System Dependencies
----|-----
Strictly speaking, you now have everything you need to run the core Caffe2 successfully. However, for real-world deep learning (e.g., image processing, mathematical operations, etc), there are other dependencies that you will want to install in order to experience the full features of Caffe2. |
[OpenCV](http://opencv.org/) for image-related operations. |
[OpenMPI](http://www.open-mpi.org/) for MPI-related Caffe2 operators. |
[RocksdB](http://rocksdb.org) for Caffe2's RocksDB IO backend. |
[ZeroMQ](http://zeromq.org/), needed for Caffe2's ZmqDB IO backend (serving data through a socket). |
[cuDNN](https://developer.nvidia.com/cudnn), if using GPU, this is needed for Caffe2's cuDNN operators. |

| Python Optional Dependencies
----|-----
There are also various Python libraries that will be valuable in your experience with Caffe2. Many of these are required to run the tutorials. |
[Flask](http://flask.pocoo.org/) |
[Jupyter](https://ipython.org/) for the Jupyter Notebook |
[Matplotlib](http://matplotlib.org/) |
[Pydot](https://pypi.python.org/pypi/pydot) |
[Python-nvd3](https://pypi.python.org/pypi/python-nvd3/) |
[SciPy](https://www.scipy.org/) |
[Tornado](http://www.tornadoweb.org/en/stable/) |
[Scikit-Image](http://scikit-image.org/) |

<block class="ubuntu prebuilt" />

### Prebuilt Binaries

** COMING SOON **

<block class="ubuntu docker" />

### Docker Images

Ubuntu docker images are currently not available. We are looking into providing them sometime in the future.
