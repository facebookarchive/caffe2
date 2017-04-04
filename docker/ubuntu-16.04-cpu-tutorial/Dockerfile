#FROM ubuntu:16.04
FROM caffe2ai/caffe2:cpu-fulloptions-ubuntu16.04
MAINTAINER Aaron Markham <aaronmarkham@fb.com>

# caffe2 install with cpu support

########## INSTALLATION STEPS ###################
RUN apt-get install unzip
WORKDIR "/"
RUN rm -rf caffe2
RUN git clone --recursive https://github.com/caffe2/caffe2.git 

########## REBUILD ###################
# rebuild is broken as of 3.30.17
WORKDIR "/caffe2"
RUN make
WORKDIR "/caffe2/build"
RUN make install

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

########## SETUP TUTORIAL FILES #################
# get squeeznet model files for Loading Pre-Trained Models
WORKDIR "/root/models/squeezenet"
RUN wget "https://s3.amazonaws.com/caffe2/models/squeezenet/init_net.pb"
RUN wget "https://s3.amazonaws.com/caffe2/models/squeezenet/predict_net.pb"
# get MNIST dataset for MNIST
WORKDIR "/caffe2/caffe2/python/tutorials"
RUN mkdir tutorial_data && cd tutorial_data
WORKDIR "/caffe2/caffe2/python/tutorials/tutorial_data"
RUN wget "https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip"
RUN unzip -d mnist mnist.zip
WORKDIR "/caffe2/caffe2/python/tutorials"

########## TROUBLESHOOTING ######################
