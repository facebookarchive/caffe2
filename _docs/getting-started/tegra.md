<block class="tegra compile" />

To install Caffe2 on NVidia's Tegra X1 platform, simply install the latest system with the [NVidia JetPack installer](https://developer.nvidia.com/embedded/jetpack), clone the Caffe2 source, and then run scripts/build_tegra_x1.sh on the Tegra device.

## Install JetPack

* [NVidia JetPack installer](https://developer.nvidia.com/embedded/jetpack)

## Download Caffe2 Source

If you have not done so already, download the Caffe2 source code from GitHub.

```
git clone --recursive https://github.com/caffe2/caffe2.git
git submodule update --init
```

## Run the Build Script

Run scripts/build_tegra_x1.sh on the Tegra device.

```
cd caffe2
./scripts/build_tegra_x1.sh
```


<block class="tegra prebuilt" />

There are no pre-built binaries available for Tegra yet. Please install from [source](https://caffe2.ai/docs/getting-started.html?platform=tegra&configuration=compile).


<block class="tegra docker" />

There are no Docker images available for Tegra. Please install from [source](https://caffe2.ai/docs/getting-started.html?platform=tegra&configuration=compile).


<block class="tegra cloud" />
