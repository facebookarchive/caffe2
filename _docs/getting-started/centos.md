{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="centos prebuilt" />

We only support Anaconda packages at the moment. If you do not wish to use Anaconda, then you must build Caffe2 from [source](https://caffe2.ai/docs/getting-started.html?platform=centos&configuration=compile).

### Anaconda packages

We build Linux packages without CUDA support, with CUDA 9.0 support, and with CUDA 8.0 support, for both Python 2.7 and Python 3.6. These packages are built on Ubuntu 16.04, but they will probably work on CentOS as well (if they do not, please tell us by creating an issue on our [Github page](https://github.com/caffe2/caffe2/issues)). To install Caffe2 with Anaconda, simply activate your desired conda environment and then run one of the following commands:

> If your gcc version is older than 5 (you can run ```gcc --version``` to find out), then append '-gcc4.8' to the package name. For example, run `conda install -c caffe2 caffe2-gcc4.8` or `conda install -c caffe2 caffe2-cuda9.0-cudnn7-gcc4.8` instead of what's below. (The command you run should always start with `conda install -c caffe2`)

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

To see what libraries these Caffe2 conda packages are built against, see `conda/cuda/meta.yaml` for the CUDA builds and `conda/no_cuda/meta.yaml` for the CPU only build. If you want to use different libraries then you must build from source.


<block class="centos docker" />

<block class="centos compile" />

Check the cloud instructions for a general guideline on building from source for CentOS.

The installation instructions for [Ubuntu](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile) will probably also work in most cases.

<block class="centos cloud" />

## AWS Cloud Setup

### Amazon Linux AMI with NVIDIA GRID and TESLA GPU Driver

[NVIDIA GRID and TESLA GPU](https://aws.amazon.com/marketplace/pp/B00FYCDDTE?qid=1489162823246&sr=0-1&ref_=srh_res_product_title)

The above AMI had been tested with Caffe2 + GPU support on a G2.2xlarge instance that uses a NVIDIA GRID K520 GPU. This AMI comes with CUDA v7.5, and no cuDNN, so we install that manually. The installation is currently a little tricky, but we hope over time this can be smoothed out a bit. This AMI is great though because it supports the [latest and greatest features from NVIDIA](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html).

### Installation Guide

Note that this guide will help you install Caffe2 on any CentOS distribution. Amazon uses their own flavor of RHEL and they've installed CUDA in different spots than normally expected, so keep that in mind if you have to do some troubleshooting. Some of these steps will not be required on vanilla CentOS because things will go in their normal places.

#### Get your repos set

Many of the required dependencies don't show up in Amazon's enabled repositories. Epel is already provided in this image, but the repo is disabled. You need to enable it by editing the repo config to turn it on. Set `enabled=1` in the `epel.repo` file. This enables you to find `cmake3 leveldb-devel lmdb-devel`.

```
sudo vim /etc/yum.repos.d/epel.repo
```

![epel repo edit](../static/images/centos-epel.png)

Next you should update yum and install Caffe2's core dependencies. These differ slightly from Ubuntu due to availability of ready-to-go packages.

```
sudo yum update
sudo yum install -y \
automake \
cmake3 \
gcc \
gcc-c++ \
git \
kernel-devel \
leveldb-devel \
lmdb-devel \
libtool \
protobuf-devel \
python-devel \
python-pip \
snappy-devel
```

gflags and glog is not found in yum for this version of Linux, so install from source:

```
git clone https://github.com/gflags/gflags.git && \
cd gflags && \
mkdir build && cd build && \
cmake3 -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS='-fPIC' .. && \
make -j 8 && sudo make install && cd ../.. && \
git clone https://github.com/google/glog && \
cd glog && \
mkdir build && cd build && \
cmake3 -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS='-fPIC' .. && \
make -j 8 && sudo make install && cd ../..
```

#### Python Dependencies

Now we need the Python dependencies. Note the troubleshooting info below... the install path with Python can get difficult.

```
sudo pip install \
future \
graphviz \
hypothesis \
jupyter \
matplotlib \
numpy \
protobuf \
pydot \
python-nvd3 \
pyyaml \
requests \
scikit-image \
scipy \
six
```

This may fail with error:
`pkg_resources.DistributionNotFound: pip==7.1.0`

To fix this, upgrade pip, and then update the pip's config to match the version it upgraded to.

```
$ sudo easy_install --upgrade pip
Password:
Searching for pip
Reading https://pypi.python.org/simple/pip/
Downloading https://pypi.python.org/packages/11/b6/abcb525026a4be042b486df43905d6893fb04f05aac21c32c638e939e447/pip-9.0.1.tar.gz#md5=35f01da33009719497f01a4ba69d63c9
Best match: pip 9.0.1
```

Note that in this example, the upgrade was to 9.0.1. Use vim to open the `/usr/bin/pip` file and change the instances of `7.1.0` to `9.0.1`, and this solves the pip error and will allow you to install the dependencies.

```
sudo vim /usr/bin/pip
```

![pip edit](../static/images/centos-pip.png)

Once you've fixed the config file re-run the `sudo pip install graphviz...` command from above.

#### Setup CUDA

This image doesn't come with cuDNN, however Caffe2 requires it. Here we're downloading the files, extracting them, and copying them into existing folders where CUDA is currently installed.

**Note: recent developments indicate that you should try to upgrade to CUDA 8 and cuDNN 6, however these instructions provide a working build with v7.5 and cuDNN 5.1.**

```
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
tar xfvz cudnn-7.5-linux-x64-v5.1.tgz
sudo rsync -av cuda /opt/nvidia/
rm cudnn-7.5-linux-x64-v5.1.tgz
rm -rf cuda
```

Now you need to setup some environment variables for the build step.

```
export CUDA_HOME=/opt/nvidia/cuda
export LD_LIBRARY_PATH=/opt/nvidia/cuda/lib64:/usr/local/bin
```

Almost done. Now you need to clone Caffe2 repo and build it (note: update the `-j8` with your system's number of processors; to check this, run `nproc` from the terminal.):

```
git clone --recursive https://github.com/caffe2/caffe2
cd caffe2 && git submodule update --init
mkdir build && cd build
cmake3 ..
sudo make -j8 install
```

#### Test it out!

To check if Caffe2 is working and it's using the GPU's try the commands below. The first will tell you success or failure, and the second should trigger the GPU and output of a bunch of arrays, but more importantly, you should see no error messages! Consult the Troubleshooting section of the docs here and for Ubuntu for some help.

```
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
python -m caffe2.python.operator_test.relu_op_test
```

**Test CUDA**

Here are a series of commands and sample outputs that you can try. These will verify that the GPU's are accessible.

```
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module 352.99 Mon Jul 4 23:52:14 PDT 2016
GCC version: gcc version 4.8.3 20140911 (Red Hat 4.8.3-9) (GCC)

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17

$ nvidia-smi -q | head

==============NVSMI LOG==============

Timestamp : Fri Mar 10 23:15:45 2017
Driver Version : 352.99

Attached GPUs : 1
GPU 0000:00:03.0
Product Name : GRID K520
Product Brand : Grid
```

That's it. You've successfully built Caffe2!

### Setting Up Tutorials & Jupyter Server

If you're running this all on a cloud computer, you probably won't have a UI or way to view the IPython notebooks by default. Typically, you would launch them locally with `ipython notebook` and you would see a localhost:8888 webpage pop up with the directory of notebooks running. The following example will show you how to launch the Jupyter server and connect to remotely via an SSH tunnel.

First configure your cloud server to accept port 8889, or whatever you want, but change the port in the following commands. On AWS you accomplish this by adding a rule to your server's security group allowing a TCP inbound on port 8889. Otherwise you would adjust iptables for this.

![security group screenshot](../static/images/security-group-jupyter.png)

Next you launch the Juypter server.

```
jupyter notebook --no-browser --port=8889
```

Then create the SSH tunnel. This will pass the cloud server's Jupyter instance to your localhost 8888 port for you to use locally. The example below is templated after how you would connect AWS, where `your-public-cert.pem` is your own public certificate and `ec2-user@super-rad-GPU-instance.compute-1.amazonaws.com` is your login to your cloud server. You can easily grab this on AWS by going to Instances > Connect and copy the part after `ssh` and swap that out in the command below.

```
ssh -N -f -L localhost:8888:localhost:8889 -i "your-public-cert.pem" ec2-user@super-rad-GPU-instance.compute-1.amazonaws.com
```

#### Troubleshooting

caffe2.python not found | You may have some PATH or PYTHONPATH issues. Add `/home/ec2-user/caffe2/build` to your path and that can take care of those problems.
error while loading shared libraries: libCaffe2_CPU.so: cannot open shared object file: No such file or directory | Try updating your LD_LIBRARY_PATH with `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
undefined reference to \`ncclReduceScatter' | This does not occur on Caffe2 building, but on linking with "libCaffe2_GPU.so" in some external projects. To solve this, you may install NCCL from its source bundled with Caffe2: (under the Caffe2 project directory) `cd third_party/nccl && make -j 8 && sudo make install`

{{ outro | markdownify }}
