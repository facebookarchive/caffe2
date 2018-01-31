---
docid: cplusplus_tutorial
title: Caffe2 with C++
layout: docs
permalink: /docs/cplusplus_tutorial.html
---

There are only a few documents that explain how to use Caffe2 with C++. In this tutorial I'll go through how to setup the properties for Caffe2 with C++ using VC++ in Windows.


## 1. Things you need to prepare
* [Visual Studio for C++ (over VC15)](https://www.visualstudio.com/downloads/)
* [Caffe2 Sources (from GitHub)](https://github.com/caffe2/caffe2)
* [Google Protocol Buffer Sources (from GitHub)](https://developers.google.com/protocol-buffers/)
* [Google Protocol Buffer Execution File for Windows (from Github)](https://github.com/google/protobuf/releases)



## 2. Our goal
- To make a simple console program that contains Caffe2 header files by using C++



## 3. Step
1. Create a new default project for a console program in VC.

2. Move your mouse on your project which is in the solution browser, and press **Property**.

3. On the left side of the property page, you can see the dropdown button named C/C++. Press it, then you can see subbutton named **General**. Press it.

4. On the right side of the page, there is a property named **Additional Including Directory**. Press the dropdown button which is on right side of the line, then press the **Edit** button which is below the line.

5. Type below directories in the textbox. In the below list, "XXX_DIRECTORY" means the name of the directory you installed program "XXX" in.

~~~
$CAFFE2_DIRECTORY\caffe2-master\
$CAFFE2_DIRECTORY\caffe2-master\caffe2\core
$PROTOBUF_DIRECTORY\protobuf-master\src
~~~

6. Go to the directory "**$CAFFE2_DIRECTORY\caffe2\proto**". In the directory there are some '.proto' files. You should generate '.cc' files and '.h' files from these files using Google Protocol Buffer Execution file. Put the exe file into same directory with '.proto' files, and in the prompt, execute the instruction **"protoc --cpp_out=./ $FILENAME.proto"**. Then you will have the new files you'll need to execute the program.

7. Make a simple C++ program, including some Caffe2 headers like **<blob.h>**. If that works you can try other tutorials from here!
