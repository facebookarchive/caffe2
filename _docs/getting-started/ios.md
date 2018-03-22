<block class="ios compile" />

## Install Caffe2 for your development platform

If you want to build Caffe2 for use on iOS, first follow the instructions to setup Caffe2 on your Mac platform using the toggler above, and then:

> Note Caffe2 for iOS can only be built on a Mac

## Xcode

If you have not installed [Xcode](https://developer.apple.com/xcode/) (because you used a prebuilt Caffe2 binary, etc.), [install](https://itunes.apple.com/us/app/xcode/id497799835) it first.

## Dependencies

Install [Automake](https://www.gnu.org/software/automake/) and [Libtool](https://www.gnu.org/software/libtool/libtool.html). This can be done on a Mac via `brew install automake libtool`.

## Download Caffe2 Source

If you have not done so already, download the Caffe2 source code from GitHub

```
git clone --recursive https://github.com/caffe2/caffe2.git
git submodule update --init
```

## Run the Build Script

```
cd caffe2
./scripts/build_ios.sh
```

<block class="ios prebuilt docker" />

There are no pre-built binaries available for iOS yet. Please install from [source](https://caffe2.ai/docs/getting-started.html?platform=ios&configuration=compile).

<block class="ios cloud" />
