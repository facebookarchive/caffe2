<block class="android compile" />

## Install Caffe2 for your development platform

If you want to build Caffe2 for use on Android, first follow the instructions to setup Caffe2 on your given development platform using the toggler above, and then:

## Android Studio

[Android Studio](https://developer.android.com/studio/index.html) will install all the necessary NDK, etc. components to build Caffe2 for Android use.

## Dependencies

Install [Automake](https://www.gnu.org/software/automake/) and [Libtool](https://www.gnu.org/software/libtool/libtool.html). This can be done on a Mac via `brew install automake libtool` or on Ubuntu via `sudo apt-get install automake libtool`.

## Download Caffe2 Source

If you have not done so already, download the Caffe2 source code from GitHub

```
git clone --recursive https://github.com/caffe2/caffe2.git
git submodule update --init
```

## Run the Build Script

If you want to build Caffe2 for Android with armeabi-v7a ABI:

```
cd caffe2
./scripts/build_android.sh
```

Or if you want to build Caffe2 for Android with arm64-v8a ABI:

```
cd caffe2
./scripts/build_android.sh -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN=clang
```

<block class="android prebuilt docker" />

There are no pre-built binaries available for Android yet. Please install from [source](https://caffe2.ai/docs/getting-started.html?platform=android&configuration=compile).

<block class="android" />
