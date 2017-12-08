ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}

# We don't use install_base.sh here, b/c we want to emulate online installation
# instructions. We also don't care about MKL, android, or alternate gcc or clang
# versions. Those should be covered in other Dockerfiles
ADD ./install_base_for_anaconda.sh install_base_for_anaconda.sh
RUN bash ./install_base_for_anaconda.sh && rm install_base_for_anaconda.sh

# Compile/install ccache for faster builds
ADD ./install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh

# Install Python
ARG PYTHON_VERSION
ADD ./install_python.sh install_python.sh
RUN bash ./install_python.sh && rm install_python.sh

# (optional) Add Jenkins user
ARG JENKINS
ADD ./add_jenkins_user.sh add_jenkins_user.sh
RUN if [ -n "${JENKINS}" ]; then bash ./add_jenkins_user.sh; fi
RUN rm add_jenkins_user.sh

# Install Anaconda
ARG ANACONDA_VERSION
ADD ./install_anaconda.sh install_anaconda.sh
RUN bash ./install_anaconda.sh && rm install_anaconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Install with online caffe2 instructions
ADD ./install_with_anaconda.sh install_with_anaconda.sh
RUN bash ./install_with_anaconda.sh && rm install_with_anaconda.sh
