{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

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

{{ outro | markdownify }}

<block class="ubuntu prebuilt" />

### Prebuilt Binaries

** COMING SOON **

<block class="ubuntu docker" />

### Docker Images

Ubuntu docker images are currently not available. We are looking into providing them sometime in the future.
