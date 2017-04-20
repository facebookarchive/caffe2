<block class="tegra compile" />

To install Caffe2 on NVidia's Tegra X1 platform, simply install the latest system with the [NVidia JetPack installer](https://developer.nvidia.com/embedded/jetpack), clone the Caffe2 source, and then run scripts/build_tegra_x1.sh on the Tegra device.

## Install JetPack

* [NVidia JetPack installer](https://developer.nvidia.com/embedded/jetpack)

## Download Caffe2 Source

If you have not done so already, download the Caffe2 source code from GitHub.

```
git clone --recursive https://github.com/caffe2/caffe2.git
```

## Run the Build Script

Run scripts/build_tegra_x1.sh on the Tegra device.

```
cd caffe2
./scripts/build_tegra_x1.sh
```

<block class="tegra docker" />

<block class="tegra cloud" />
