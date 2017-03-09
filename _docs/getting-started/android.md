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
```

## Run the Build Script

```
cd caffe2
./scripts/build_android.sh
```

<block class="android prebuilt docker" />

Caffe2 for Android must be built using the Android build script.

<block class="android cloud" />
