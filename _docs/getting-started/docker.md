{% capture outro %}{% include_relative getting-started/outro.md %}{% endcapture %}

{{ outro | markdownify }}

<block class="docker mac windows ubuntu" />

## Docker Images

Docker images are currently in testing. If you would like to build an image yourself, follow the instructions further below. For a quick install try the following commands (assuming you have [Docker installed](https://www.docker.com/products/overview) already).

[USB/offline or Quickstart instructions](docker-setup)

### Get caffe2ai/caffe2

Visit our [Docker repo](https://hub.docker.com/r/caffe2ai/caffe2) for a full list of different Docker options. Currently we have CPU and GPU support for both 14.04 and 16.04 Ubuntu. 

**If you wish to use GPU with Docker use `nvidia-docker` to run your image instead of regular `docker`.**
You can [get nvidia-docker here](https://github.com/NVIDIA/nvidia-docker).

For the latest Docker image using GPU support and optional dependencies like IPython & OpenCV:

```
docker pull caffe2ai/caffe2
# to test
nvidia-docker run -it caffe2ai/caffe2:latest python -m caffe2.python.operator_test.relu_op_test
# to interact
nvidia-docker run -it caffe2ai/caffe2:latest /bin/bash
```

For a minimal image:

```
docker pull caffe2ai/caffe2:cpu-minimal-ubuntu14.04 
# to test
docker run -it caffe2ai/caffe2:cpu-minimal-ubuntu14.04 python -m caffe2.python.operator_test.relu_op_test
# to interact
docker run -it caffe2ai/caffe2:cpu-minimal-ubuntu14.04 /bin/bash
```

[Caffe2 Docker Images](https://hub.docker.com/r/caffe2ai/caffe2/tags/)

See below for instructions on usage.

### Build From Dockerfile

Inside repo's `/docker` folder are subfolders with a `Dockerfile` that contain the minimal dependencies and optional ones. You may remove specific optional dependencies if you wish. The folder's name describes the defaults that will be installed by that dockerfile. For example, if you run the command below from the `ubuntu-14.04-cpu-all-options` folder you will get a docker image around 1.5GB that has many optional libraries like OpenCV, for the minimal install, `ubuntu-14.04-cpu-minimal`, it is about 1GB and is just enough to run Caffe2, and finally for the gpu dockerfile, `ubuntu-14.04-gpu-all-options`, it is based on the NVIDIA CUDA docker image about 3.2GB and contains all of the optional dependencies.

In a terminal window in one of those folders, simply run the following:

```
cd ~/caffe2/docker/ubuntu-14.04-cpu-all-options
docker build -t caffe2:cpu-optionals .
```

Don't miss the `.` as it is pointing to the `Dockerfile` in your current directory. Also, you can name docker image whatever you want. The `-t` denotes tag followed by the repository name you want it called, in this case `cpu-optionals`.

Once the build process is complete you can run it by its name or by the last unique ID that was provided upon completion. In this example case, this ID is `5ee1fb669aef`. To run the image in a container and get to bash you can launch it interactively using the following where you call it by its repository name:

```
docker run -it caffe2 /bin/bash
```

If you decide to try out the different Docker versions of Caffe2 using different dependencies then you will want to build them with their own tag and launch them using their tag or unique ID instead, for example using an ID from the previous step:

```
docker run -it 5ee1fb669aef /bin/bash
```

Or, building with a tag and then launching with the tag:

```
docker build -t caffe2:cpu-minimal .
docker run -it caffe2:cpu-minimal /bin/bash
```

### Using A Caffe2 Docker Image

[DockerHub Caffe2 Repo](https://hub.docker.com/r/caffe2ai/caffe2)

You can run specific Caffe2 commands by logging into bash as shown above, hitting the Python interface directly, or by interacting with IPython as shown below.

The simplest test was already run during the build, but you can run it again.

**Be warned that these tests only work with the optional dependencies images.**

```
docker run -it mydocker-repo/mytag ipython
```

For GPU support, use `nvidia-docker`. There's also this alternative, manual approach were you will need to pass in several device parameters. Be warned that [Windows support for this is limited](https://github.com/NVIDIA/nvidia-docker/issues/197).

```
sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm mydocker-repo/mytag ipython
```

Then once in the IPython environment you can interact with Caffe2.

```python
In [1]: from caffe2.python import workspace
```

If you want to get in the image and play around with Python or C++ directly, just launch bash like so:

```
docker run -it mydocker-repo/mytag /bin/bash
```

Another test that you can use to put Caffe2 through its paces, including GPU support, is by calling one of the [operator tests](https://github.com/caffe2/caffe2/blob/master/caffe2/python/operator_test/relu_op_test.py). Here's a [sample output](https://gist.github.com/aaronmarkham/dcdb284065c9ea4569214bcb0ca3a858).

```
nvidia-docker run -it caffe2 python -m caffe2.python.operator_test.relu_op_test
```

You may also try fetching some models directly and running them as described in this [Tutorial](../tutorials/Loading_Pretrained_Models.ipynb).

### Jupyter from Docker

If you want to run your Jupyter server from a Docker container, then you'll need to run the container with several additional flags. The first new one (versus running it locally) for Docker is `-p 8888:8888` which "publishes" the 8888 port on the container and maps it to your host's 8888 port. You also need to launch jupyter with `--ip 0.0.0.0` so that you can hit that port from your host's browser, otherwise it will only be available from within the container which isn't very helpful. Of course you'll want to swap out the `caffe2ai/caffe2:cpu-fulloptions-ubuntu14.04` with your own repo:tag for the image you want to launch.


> In this case we're running jupyter with `sh -c`. This solves a problem with the Python kernel crashing constantly when you're running notebooks.


```
docker run -it -p 8888:8888 caffe2ai/caffe2:cpu-fulloptions-ubuntu14.04 sh -c "jupyter notebook --no-browser --ip 0.0.0.0 /caffe2_tutorials"
```

Your output will be along these lines below. You just need to copy the provided URL/token combo into your browser and you should see the folder with tutorials. Note the if you installed caffe2 in a different spot, then update the optional path that is in the command `/caffe2_tutorials` to match where the tutorials are located.

![jupyter docker launch screenshot](../static/images/jupyter-docker-launch.png)

> In some situations you can't access the Jupyter server on your browser via 0.0.0.0 or localhost. You need to pull the Docker IP address (run `docker-machine ip`) and use that to access the Jupyter server.

**Docker - Ubuntu 14.04 with full dependencies notes:**

- librocksdb-dev not found. (May have to install this yourself if you want it.)

| Troubleshooting
----|-----
common_gpu.cc:42 | Found an unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. I will set the available devices to be zero.
Solution | This may be a Docker-specific error where you need to launch the images while passing in GPU device flags: `sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm mydocker-repo/mytag /bin/bash`. You will need to update those devices according to your hardware (however this should match a 1-GPU build) and you need to swap out `mydocker-repo/mytag` with the ID or the repo/tag of your Docker image.
HyperV is not available on Home editions. Please use Docker Toolbox. | Docker for Windows only works on Professional versions of Windows.
Solution | Install [Docker Toolbox](https://www.docker.com/products/docker-toolbox). Don't worry, the Caffe2 images should still work for you!
An error occurred trying to connect... | various errors just after installing Docker Toolbox...
Solution | run `docker-machine env default` then follow the instructions... run each of the commands that setup the docker environment then try `docker version` and you shouldn't see the errors again and will be able to `docker pull caffe2ai/caffe2`.
