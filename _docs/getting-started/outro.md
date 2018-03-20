
| Caffe2 Python
----|-----
Module not found | Verify that Caffe2 was installed correctly
Solution | Run the following: `python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"` An output of `Success` means you are ready to with Caffe2 - congratulations! An output of `Failure` usually means you have not installed one of the dependencies correctly.
Dependencies missing | It's possible you're trying to run something that was using an optional dependency.
Solution | `sudo pip install setuptools flask jupyter matplotlib scipy pydot tornado python-nvd3 scikit-image pyyaml`
matplotlib error | Sometimes you need setuptools first: `sudo pip install -U pip setuptools && sudo pip install matplotlib`
model downloader error | If you need to run it as sudo (because it's trying to install the model in /usr/local/caffe2...), then PYTHONPATH might not be visible in that context.
Solution | `sudo visudo` then add this line: `Defaults    env_keep += "PYTHONPATH"`
"AttributeError: 'module' object has no attribute 'MakeArgument'" | Occurs when calling `core.CreateOperator`
Solution | Check your install directory (`/usr/local/`), and remove the folder `/caffe2/python/utils`

| Other Errors
----|-----
libdc1394 error | for some reason once opencv is installed you may get [errors with libdc1394](http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394)
Solution | `ln /dev/null /dev/raw1394` but that's not [persistent](http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu) so try `sh -c 'ln -s /dev/null /dev/raw1394'` or when instantiating the container use: `--device /dev/null:/dev/raw1394`
caffe2_pybind11_state_gpu | WARNING:root:Debug message: No module named caffe2_pybind11_state_gpu
Solution | ignore if you're using CPU-only
Python kernel crashing | This happens when you try to call Jupyter server directly (like in a Docker container).
Solution | Use `sh -c "jupyter notebook ..."` to get around this problem.
Exception: "dot" not found in path | This happens in some of the tutorials when graphing.
Solution | Make sure you have `graphviz` and `pydot`. `sudo apt-get install python-pydot` and `sudo pip install graphviz` or `brew install` these to fix the problem.

### Dependencies

Try to keep the system and python dependencies at the same version. We've encountered [issues](https://github.com/caffe2/caffe2/issues) when the python version is more updated than the system version or vice versa.


| System Dependencies
----|-----
[cmake](https://cmake.org/download/) |
[git](https://github.com) |
[gflags](https://gflags.github.io/gflags/) |
[glog: Google Logging Module](https://github.com/google/glog) |
[NumPy](http://www.numpy.org/) |
[protobuf: Google Protocol Buffers](https://developers.google.com/protocol-buffers/) | version 3.2.0
Build tools for C++ 11 | Xcode CLTs & automake (mac/iOS), build-essential (linux), Visual Studio (win), Android Studio (droid)


| Python Dependencies
----|-----
[gflags](https://gflags.github.io/gflags/) |
[glog: Google Logging Module](https://github.com/google/glog) |
[NumPy](http://www.numpy.org/) |
[protobuf: Google Protocol Buffers](https://developers.google.com/protocol-buffers/) | version 3.2.0


Strictly speaking, the core dependencies above are all you need to run the core Caffe2 successfully. However, for real-world deep learning (e.g., image processing, mathematical operations, etc), there are other dependencies that you will want to install in order to experience the full features of Caffe2.


| Optional System Dependencies
----|-----
[cuDNN](https://developer.nvidia.com/cudnn) | if using GPU, this is needed for Caffe2's cuDNN operators
[Eigen 3](http://eigen.tuxfamily.org/) | The default BLAS backend
[LevelDB](https://github.com/google/leveldb) | One of the DB options for storing Caffe2 models
[Nvidia CUDA](https://developer.nvidia.com/cuda-zone) | v6.5 or greater
[OpenCV](http://opencv.org/) | for image-related operations; requires leveldb <= v1.19
[OpenMPI](http://www.open-mpi.org/) | for MPI-related Caffe2 operators, which are used for distributed training
[RocksdB](http://rocksdb.org) | for Caffe2's RocksDB IO backend
[ZeroMQ](http://zeromq.org/) | needed for Caffe2's ZmqDB IO backend (serving data through a socket)


| Python Optional Dependencies
----|-----
There are also various Python libraries that will be valuable in your experience with Caffe2. Many of these are required to run the tutorials. |
[Graphviz](http://www.graphviz.org/) | Used for plotting in the Jupyter Notebook Tutorials
[Hypothesis](https://hypothesis.readthedocs.io/) | Used in all of the tests
[Jupyter](https://ipython.org/) | Used for interactive python notebooks.
[LevelDB](https://github.com/google/leveldb) | One of the DB options for storing Caffe2 models
[lmdb](https://lmdb.readthedocs.io/en/release/) | One of the DB options for storing Caffe2 models
[Matplotlib](http://matplotlib.org/) | Used for plotting in the Jupyter Notebook Tutorials
[Pydot](https://pypi.python.org/pypi/pydot) | Used for plotting in the Jupyter Notebook Tutorials
[Python-nvd3](https://pypi.python.org/pypi/python-nvd3/) |
[pyyaml](http://pyyaml.org/) | Used in the MNIST tutorial
[requests](http://docs.python-requests.org/en/master/) | Used in the MNIST tutorial to download the dataset
[Scikit-Image](http://scikit-image.org/) |
[SciPy](https://www.scipy.org/) |
[ZeroMQ](http://zeromq.org/) | needed for Caffe2's ZmqDB IO backend (serving data through a socket)
