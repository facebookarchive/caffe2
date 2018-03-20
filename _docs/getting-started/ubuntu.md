{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="ubuntu prebuilt" />

We only support Anaconda packages at the moment. If you do not wish to use Anaconda, then you must build Caffe2 from [source](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile).

### Anaconda packages

We build Linux packages without CUDA support, with CUDA 9.0 support, and with CUDA 8.0 support, for both Python 2.7 and Python 3.6. These packages are built on Ubuntu 16.04, but they will probably work on Ubuntu14.04 as well (if they do not, please tell us by creating an issue on our [Github page](https://github.com/caffe2/caffe2/issues)). To install Caffe2 with Anaconda, simply activate your desired conda environment and then run one of the following commands:

> If your gcc version is older than 5 (you can run ```gcc --version``` to find out), then append '-gcc4.8' to the package name. For example, run `conda install -c caffe2 caffe2-gcc4.8` instead of what's below. (The command you run should always start with `conda install -c caffe2`)

If you do not have a GPU:

```bash
conda install -c caffe2 caffe2
```

For GPU support you will need [CUDA](https://developer.nvidia.com/cuda-downloads), [CuDNN](https://developer.nvidia.com/cudnn), and [NCCL](https://developer.nvidia.com/nccl). These must be installed from Nvidia's website. 

For Caffe2 with CUDA 9 and CuDNN 7 support:

```bash
conda install -c caffe2 caffe2-cuda9.0-cudnn7
```

For Caffe2 with CUDA 8 and CuDNN 7 support:

```bash
conda install -c caffe2 caffe2-cuda8.0-cudnn7
```

> This does NOT include libraries that are necessary to run the tutorials, such as jupyter. See the [tutorials](https://caffe2.ai/docs/tutorials) page for the list of required packages needed to run the tutorials.

NOTE: This will install Caffe2 and all of its required dependencies into the current conda environment. We strongly suggest that you create a new conda environment and install Caffe2 into that. A conda environment is like its own python installation that won't have library version problems with your other conda environments. You can learn more about conda environments [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

To see what libraries these Caffe2 conda packages are built against, see `conda/cuda/meta.yaml` for the CUDA builds and `conda/no_cuda/meta.yaml` for the CPU only build. If you want to use different libraries then you must build from  source.


<block class="ubuntu cloud" />
You can easily try out Caffe2 by using the Cloud services. Caffe2 is available as AWS (Amazon Web Services) Deep Learning AMI and Microsoft Azure Virtual Machine offerings. You can run run Caffe2 in the Cloud at any scale.

* [AWS Deep Learning AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B06VSPXKDX?qid=1489099515180&sr=0-6&ref_=srh_res_product_title)
* [Microsoft Azure Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu?tab=Overview)

<block class="ubuntu compile" />

This build is confirmed for:

* Ubuntu 14.04
* Ubuntu 16.04

> Anaconda users: To build with Anaconda, follow the instructions on the [Mac page](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=compile#anaconda-install-path).

### Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip                          
sudo pip install \
      future \
      numpy \
      protobuf
```

> Note `libgflags2` is for Ubuntu 14.04. `libgflags-dev` is for Ubuntu 16.04.

```bash
# for Ubuntu 14.04
sudo apt-get install -y --no-install-recommends libgflags2
# for Ubuntu 16.04
sudo apt-get install -y --no-install-recommends libgflags-dev
```

> If you have a GPU, follow [these additional steps](#install-with-gpu-support) before continuing.

### Clone & Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2

# This will build Caffe2 in an isolated directory so that Caffe2 source is
# unaffected
mkdir build && cd build

# This configures the build and finds which libraries it will include in the
# Caffe2 installation. The output of this command is very helpful in debugging
cmake ..

# This actually builds and installs Caffe2 from makefiles generated from the
# above configuration step
sudo make install
```

### Test the Caffe2 Installation
Run this to see if your Caffe2 installation was successful. 

```bash
cd ~ && python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

If this fails, then get a better error message by running a Python interpreter in the `caffe2/build` directory and then trying `from caffe2.python import core`.

If this fails with a message about not finding caffe2.python or not finding libcaffe2.so, [check your environment variables](#environment-variables) first.

If you installed with GPU support, test that the GPU build was a success with this command. You will get a test output either way, but it will warn you at the top of the output if CPU was used instead of GPU, along with other errors such as missing libraries.

```bash
python caffe2/python/operator_test/relu_op_test.py
```

<block class="ubuntu compile" />
### Install with GPU Support

If you plan to use GPU instead of CPU only, then you should install NVIDIA CUDA 8 and cuDNN v5.1 or v6.0, a GPU-accelerated library of primitives for deep neural networks.
[NVIDIA's detailed instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) or if you're feeling lucky try the quick install set of commands below.

**Update your graphics card drivers first!** Otherwise you may suffer from a wide range of difficult to diagnose errors.

**For Ubuntu 14.04**

```bash
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

**For Ubuntu 16.04**

```bash
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

#### Install cuDNN (all Ubuntu versions)

**Version 5.1**
```
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```

**Version 6.0**
Visit [NVIDIA's cuDNN download](https://developer.nvidia.com/rdp/cudnn-download) to register and download the archive. Follow the same instructions above switching out for the updated library.

### Environment Variables

These environment variables may assist you depending on your current configuration. When using the install instructions above on the AWS Deep Learning AMI you don't need to set these variables. However, our Docker scripts built on Ubuntu-14.04 or NVIDIA's CUDA images seem to benefit from having these set. If you ran into problems with the build tests above then these are good things to check. Echo them first and see what you have and possibly append or replace with these directories. Also visit the [Troubleshooting](getting-started.html#troubleshooting) section.

```bash
echo $PYTHONPATH
# export PYTHONPATH=/usr/local:$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/caffe2/build
echo $LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Setting Up Tutorials & Jupyter Server

If you're running this all on a cloud computer, you probably won't have a UI or way to view the IPython notebooks by default. Typically, you would launch them locally with `ipython notebook` and you would see a localhost:8888 webpage pop up with the directory of notebooks running. The following example will show you how to launch the Jupyter server and connect to remotely via an SSH tunnel.

First configure your cloud server to accept port 8889, or whatever you want, but change the port in the following commands. On AWS you accomplish this by adding a rule to your server's security group allowing a TCP inbound on port 8889. Otherwise you would adjust iptables for this.

![security group screenshot](../static/images/security-group-jupyter.png)

Next you launch the Juypter server.

```
jupyter notebook --no-browser --port=8889
```

Then create the SSH tunnel. This will pass the cloud server's Jupyter instance to your localhost 8888 port for you to use locally. The example below is templated after how you would connect AWS, where `your-public-cert.pem` is your own public certificate and `ubuntu@super-rad-GPU-instance.compute-1.amazonaws.com` is your login to your cloud server. You can easily grab this on AWS by going to Instances > Connect and copy the part after `ssh` and swap that out in the command below.

```
ssh -N -f -L localhost:8888:localhost:8889 -i "your-public-cert.pem" ubuntu@super-rad-GPU-instance.compute-1.amazonaws.com
```

### Troubleshooting

|Python errors
----|-----
Python version | [Python](https://www.python.org/) is core to run Caffe2. We currently require [Python2.7](https://www.python.org/download/releases/2.7/). *Ubuntu 14.04 and greater have Python built in by default*, and that can be used to run Caffe2. To check your version: `python --version`
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
libgflags2 error | This optional dependency is for Ubuntu 14.04.
Solution | Use `apt-get install libgflags-dev` for Ubuntu 16.04.

| GPU Support
----|-----
GPU errors | Unsupported GPU or wrong version
Solution | You need to know the specific `deb` for your version of Linux. `sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb` Refer to NVIDIA's [installation guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).
Build issues | Be warned that installing CUDA and cuDNN will increase the size of your build by about 4GB, so plan to have at least 12GB for your Ubuntu disk size.

{{ outro | markdownify }}

<block class="ubuntu docker" />

<block class="ubuntu cloud" />
