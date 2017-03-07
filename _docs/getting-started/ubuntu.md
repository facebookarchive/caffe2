{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

<block class="ubuntu compile" />

[![Build Status](https://travis-ci.org/caffe2/caffe2.svg?branch=master)](https://travis-ci.org/caffe2/caffe2)

### Required Dependencies

```bash
sudo apt-get update
sudo apt-get install python-dev python-pip git build-essential cmake libprotobuf-dev protobuf-compiler libgoogle-glog-dev
sudo pip install numpy protobuf
```

### Optional GPU Support

If you plan to use GPU instead of CPU only, then you should install NVIDIA CUDA and cuDNN, a GPU-accelerated library of primitives for deep neural networks.
[NVIDIA's detailed instructions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation) or if you're feeling lucky try the quick install set of commands below.

**Update your graphics card drivers first!** Otherwise you may suffer from a wide range of difficult to diagnose errors.

```bash
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
sudo apt-get install cuda
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```

### Optional Dependencies

```bash
sudo apt-get install libgtest-dev libgflags2 libgflags-dev liblmdb-dev libleveldb-dev libsnappy-dev libopencv-dev libiomp-dev openmpi-bin openmpi-doc libopenmpi-dev python-pydot
sudo pip install setuptools hypothesis flask graphviz jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml
```

* Note for Ubuntu 16.04 `libgflags2` should be replaced with `libgflags-dev`.

### Clone & Build

```bash
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make && cd build && sudo make install
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

Run this command below to test if your GPU build was a success. You will get a test output either way, but it will warn you at the top of the output if CPU was used instead along with other errors like missing libraries.

```bash
python -m caffe2.python.operator_test.relu_op_test
```

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

#### Jupyter from Docker

If you want to do something similar, but run your Jupyter server from a Docker container, then you'll need to run the container with a few more flags. The first new one for Docker is `-p 8888:8888` which "publishes" the 8888 port on the container and maps it to your host's 8888 port. You also need to launch jupyter with `--ip 0.0.0.0` so that you can hit that port from your host's browser, otherwise it will only be available from within the container which isn't very helpful. Of course you'll want to swap out the `caffe2ai/caffe2:cpu-fulloptions-ubuntu14.04` with your own repo:tag for the image you want to launch.

Note: in this case we're running jupyter with `sh -c`. This solves a problem with the Python kernel crashing constantly when you're running notebooks.

```
docker run -it -p 8888:8888 caffe2ai/caffe2:cpu-fulloptions-ubuntu14.04 sh -c "jupyter notebook --no-browser --ip 0.0.0.0 /caffe2/caffe2/python/tutorials"
```

Your output will be along these lines below. You just need to copy the provided URL/token combo into your browser and you should see the folder with tutorials. Note the if you installed caffe2 in a different spot, then update the optional path that is in the command `/caffe2/caffe2/python/tutorials` to match where the tutorials are located.

![jupyter docker launch screenshot](../static/images/jupyter-docker-launch.png)

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
common_gpu.cc:42 | Found an unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. I will set the available devices to be zero.
Solution | This may be a Docker-specific error where you need to launch the images while passing in GPU device flags: `sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm mydocker-repo/mytag /bin/bash`. You will need to update those devices according to your hardware (however this should match a 1-GPU build) and you need to swap out `mydocker-repo/mytag` with the ID or the repo/tag of your Docker image.

{{ outro | markdownify }}

<block class="ubuntu prebuilt" />

### Prebuilt Binaries

** COMING SOON **

<block class="ubuntu docker" />

### Docker Images

Refer to the Mac --> Docker option on the top of this Install page to see Docker images and build options.
