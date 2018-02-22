# Caffe2 - ARM Compute Backend

## Build

To build, clone and install scons

```
brew install scons
```

use the build\_android.sh:

```
./scripts/build_android.sh -DUSE_ARM_COMPUTE=ON
```

## Test
Plug in an android device, and run a test

```
cd build_android
adb push bin/model_test /data/local/tmp && adb shell '/data/local/tmp/model_test'
```
or use a script to run them all

In caffe2 top level directory
```
./caffe2/mobile/contrib/arm-compute/run_tests.sh build_android
```