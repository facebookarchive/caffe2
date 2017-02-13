:: #############################################################################
:: Example command to build on Windows.
:: #############################################################################

:: This script shows how one can build a Caffe2 binary for windows.

@echo off

SET ORIGINAL_DIR=%cd%
SET CAFFE2_ROOT=%~dp0%..
if not exist %CAFFE2_ROOT%\build_host_protoc\bin\protoc.exe %CAFFE2_ROOT%\scripts\build_host_protoc.bat

if not exist %CAFFE2_ROOT%\build mkdir %CAFFE2_ROOT%\build
cd %CAFFE2_ROOT%\build

cmake .. ^
  -DCMAKE_VERBOSE_MAKEFILE=1 ^
  -DBUILD_TEST=OFF ^
  -DUSE_CUDA=OFF ^
  -DUSE_NCCL=OFF ^
  -DPROTOBUF_PROTOC_EXECUTABLE=%CAFFE2_ROOT%\build_host_protoc\bin\protoc.exe

cd %ORIGINAL_DIR%