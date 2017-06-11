---
docid: cplusplus_tutorial
title: Getting start Caffe2 with C++
layout: docs
permalink: /docs/cplusplus_tutorial.html
---

# A Basic Console Program by C++ using Caffe2
There are only few documents which explain how to apply Caffe2 using C++. So in this document, I'll introduce the way to prepare to make a caffe2 program: how to set the properties of the program for Caffe2 using C++, especially VC++ in windows. You can approach other tutorials from sample programs which will be introduced in this this document.


## 1. Things you need to prepare
* [Visual Studio for C++ (over VC15)](https://www.visualstudio.com/downloads/)
* [Caffe2 Sources (from GitHub)](https://github.com/caffe2/caffe2)
* [Google Protocol Buffer Sources (from GitHub)](https://developers.google.com/protocol-buffers/)
* [Google Protocol Buffer Execution File for Windows (from Github)](https://github.com/google/protobuf/releases)



## 2. Our goal
- To make a simplest console program which contains some caffe2 header file by using C++



## 3. Step
1. Make a new project for developing console program in VC, with default project.

2. Move your mouse on your project which is in the solution browser, and press **property**.

3. On the left side of the property page, you can see the dropdown button named C/C++. Press it, then you can see subbutton named **general**. Press it.

4. On the right side of the page, there is a property named **additional including directory**. Press the dropdown button which is on right side of the line, then press the **edit** button which is below the line.

5. Type below directories in the textbox. In the below directories, "XXX_DIRECTORY" means the name of the directory you installed program "XXX" in.

~~~
$CAFFE2_DIRECTORY\caffe2-master\
$CAFFE2_DIRECTORY\caffe2-master\caffe2\core
$PROTOBUF_DIRECTORY\protobuf-master\src
~~~

6. Go to the directory "**$CAFFE2_DIRECTORY\caffe2-master\caffe2\proto**". In the directory there are some '.proto' files. You should generate '.cc' files and '.h' files from these files using Google Protocol Buffer Execution file. Put the exe file into same directory with '.proto' files, and in the prompt, execute the instruction **"protoc --cpp_out=./ $FILENAME.proto"**. Then you can get new file that you need to execute the program.

7. Make a simple c++ program, including some caffe2 headers like **<blob.h>**. Does it run well? If it does, you can try any other tutorials from here. 



## Change Log
* 17.5.2013 : First version of the tutorial document.
