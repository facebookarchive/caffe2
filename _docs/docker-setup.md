---
docid: docker-setup
title: Running Caffe2 from a Docker Image
layout: docs
permalink: /docs/docker-setup.html
---

## Quickstart (Feeling Lucky)

**Assumes you have Docker and are using a Mac**

```
docker load -i /Volumes/CAFFE2/c2.gpu.tutorial.0.7.0.tar
docker run -it -p 8888:8888 cc2.gpu.tutorial.0.7.0 sh -c "jupyter notebook --no-browser --ip 0.0.0.0 /caffe2_tutorials"
```

Essentially you need to locate the tar file, whatever its name is and import it with `docker load -i <path-to-image-tar-file>`
Windows users: you can just change "/Volumes" to "D:\" or whatever the drive letter the USB was assigned and it should work.

## Setup Docker

You'll need Docker installed on your local PC. Skip ahead if you've done this already. To be sure run this command:

```
docker version
```

### Mac

* [Install Instructions](https://docs.docker.com/docker-for-mac/install/)
* [Binary (online)](https://download.docker.com/mac/stable/Docker.dmg)
* [Local Binary (if on USB)](Docker.dmg)

### Windows

* [Install Instructions](https://docs.docker.com/docker-for-windows/install/)
* [Binary (online)](https://download.docker.com/win/stable/InstallDocker.msi)
* [Local Binary (if on USB)](InstallDocker.msi)

## Get Caffe2 Docker Image

You have two ways to do this. If you're running this from a USB stick, then continue, if not, jump to the online option below.

### Local/USB: Import the Caffe2 Docker Image

This image is in a tar file on the USB stick. You can import it by using this command:

```
docker load -i <path-to-image-tar-file>
```

### Online: Pull the Caffe2 Docker Image

For the latest Docker image using GPU support and optional dependencies like IPython & OpenCV (don't bother on Windows - see Troubleshooting notes):

```
docker pull caffe2ai/caffe2 && docker run -it caffe2ai/caffe2:latest
```

## Launch the Image with Jupyter Notebook

Once the loading of the image finishes, check your list of Docker images:

```
docker images
```

Assuming it's there you can now launch it:

```
docker run -it -p 8888:8888 c2.gpu.tutorial.0.7.0 sh -c "jupyter notebook --no-browser --ip 0.0.0.0 /caffe2_tutorials"
```

This will output a URL. You just need to copy the provided URL/token combo into your browser and you should see the folder with tutorials.

> In some situations you can't access the Jupyter server on your browser via 0.0.0.0 or localhost. You need to pull the Docker IP address (run `docker-machine ip`) and use that to access the Jupyter server. If this doesn't work, check your computer's IP address and try that. If that doesn't work, kill the server, start docker-machine as mentioned in troubleshooting, check its IP, then start the Jupyter server and use the docker-machine IP.

## Using Docker and GPUs

To enable the power of your GPU while using Docker, you will want to install NVIDIA's [nvidia-docker](https://devblogs.nvidia.com/parallelforall/nvidia-docker-gpu-server-application-deployment-made-easy/). Use `nvidia-docker run ...` instead of `docker run...`.

### Troubleshooting

Getting Docker to run after installation may take some prodding and setting up of the environment. Try this:

```
docker-machine restart default
eval $(docker-machine env default)
```

More info on this setup is found on the Caffe2 docs site in Install>OS>Docker and in Install>OS>Cloud. The Cloud page has info specific to forwarding Docker through your SSH tunnel to your cloud server.

| Troubleshooting
----|-----
common_gpu.cc:42 | Found an unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. I will set the available devices to be zero.
Solution | This may be a Docker-specific error where you need to launch the images while passing in GPU device flags: `sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm mydocker-repo/mytag /bin/bash`. You will need to update those devices according to your hardware (however this should match a 1-GPU build) and you need to swap out `mydocker-repo/mytag` with the ID or the repo/tag of your Docker image.
HyperV is not available on Home editions. Please use Docker Toolbox. | Docker for Windows only works on Professional versions of Windows.
Solution | Install [Docker Toolbox](https://www.docker.com/products/docker-toolbox). Don't worry, the Caffe2 images should still work for you!
An error occurred trying to connect... | various errors just after installing Docker Toolbox...
Solution | run `docker-machine env default` then follow the instructions... run each of the commands that setup the docker environment then try `docker version` and you shouldn't see the errors again and will be able to `docker pull caffe2ai/caffe2`.
