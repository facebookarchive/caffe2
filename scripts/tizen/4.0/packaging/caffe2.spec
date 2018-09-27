Name:           caffe2
Version:        0.7.0
Release:        0
Summary:        A deep learning, cross platform ML framework with the flexibility of Python and the speed of C++.
Group:          Development/Tools
License:        Apache-2.0
URL:            https://github.com/caffe2/caffe2
Distribution:   Tizen
Vendor:         Samsung Electronics
Packager:       Geunsik Lim <geunsik.lim@samsung.com>
Source0:        %{name}-%{version}.tar.gz
Source1001:     %{name}.manifest
Patch0:         %{name}-eigen3-path.patch

BuildRequires:  make cmake gcc binutils glibc glibc-devel cpp libstdc++
BuildRequires:  protobuf-devel
BuildRequires:  eigen3-devel
BuildRequires:  openblas-devel
BuildRequires:  lmdb-devel
BuildRequires:  leveldb-devel
BuildRequires:  gflags-devel
BuildRequires:  boost-devel
BuildRequires:  opencv-devel mesa

BuildRequires:  python-python-gflags
BuildRequires:  python = 2.7
BuildRequires:  python-numpy-devel >= 1.7
BuildRequires:  python-pybind11-devel

Requires:       libleveldb
Requires:       python-pybind11
Requires:       opencv

# Note that do not add git package to keep the build structure of GBS/OBS
# BuildRequires:  git

%description
Caffe2 is a deep learning framework that provides an easy and straightforward
way for you to experiment with deep learning and leverage community contributions
of new models and algorithms. You can bring your creations to scale using the
power of GPUs in the cloud or to the masses on mobile with cross-platform
libraries of Caffe2.

%package devel
Summary:	caffe2 development package
Group:		Development/Libraries/C and C++
Provides:	caffe-devel = %{version}-%{release}
Requires:	%{name} = %{version}
Requires:	glibc-devel
Requires:	gflags-devel
Requires:	openblas-devel
Requires:	protobuf-devel
Requires:	glog-devel

%description devel
Header files and documentation for %{name} development.

%package python
Summary:	caffe2 python libraries
Group:		Development/Libraries/C and C++
Provides:	caffe-devel = %{version}-%{release}
Requires:	%{name} = %{version}
Requires:	libprotobuf

%description python
python libraries for %{name}.

%prep
%setup -q
cp %{SOURCE1001} .
%patch0 -p1 -b .path

%build

mkdir build
pushd ./build
# Note: add more dependencies above if you need libraries such as leveldb, lmdb, and so on.
# If you have to disable a specific package due to a package absence
# from https://git.tizen.org/cgit/, append -Dxxx_xxx=OFF option before executing cmake.
#  -DUSE_BENCHMARK=OFF \
#  -DCMAKE_INSTALL_PREFIX:PATH=/home/abuild/caffe2_deploy ..
cmake \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DBUILD_TEST=OFF \
    -DUSE_GFLAGS=ON  \
    -DUSE_GLOG=ON \
    -DUSE_NNPACK=OFF \
    -DRUN_HAVE_STD_REGEX=0 \
    -DRUN_HAVE_POSIX_REGEX=0 \
    -DHAVE_GNU_POSIX_REGEX=0 \
    -DUSE_MPI=OFF \
    -DUSE_OPENMP=ON \
    -DUSE_ROCKSDB=OFF \
    -DUSE_LEVELDB=ON \
    -DUSE_LMDB=ON \
    -DBUILD_PYTHON=ON \
    -DUSE_GLOO=OFF \
    -DUSE_OPENCV=ON \
    -DUSE_CUDA=OFF \
%ifarch armv7l
    -DCAFFE2_CPU_FLAGS="-mfpu=neon -mfloat-abi=soft" \
%endif
%ifarch aarch64
    -DCAFFE2_CPU_FLAGS="-mtune=cortex-a53 -mabi=lp64" \
%endif
    .. \
    || exit 1
echo -e "Building Caffe2"
make %{?_smp_mflags} || exit 1

# verify a share file that is generated for Caffe2/CPU
size ./caffe2/libCaffe2_CPU.so 
size --format=sysv ./caffe2/libCaffe2_CPU.so 
popd

%install
pushd build
# create folders
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_libdir}
mkdir -p %{buildroot}%{_includedir}/caffe2/core/
mkdir -p %{buildroot}%{_includedir}/caffe2/proto/
mkdir -p %{buildroot}%{_includedir}/caffe/proto/
mkdir -p %{buildroot}%{python_sitelib}/caffe2/

# binary tools
install -m 0755 -p ./caffe2/binaries/convert_caffe_image_db %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/convert_db %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/db_throughput %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/make_cifar_db %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/make_mnist_db %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/predictor_verifier %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/print_registered_core_operators %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/run_plan %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/speed_benchmark %{buildroot}%{_bindir}
install -m 0755 -p ./caffe2/binaries/split_db %{buildroot}%{_bindir}

# for CPU
# TODO: support another architecture such as GPU, NPU.
install -m 0644 -p ./caffe2/libCaffe2_CPU.so %{buildroot}%{_libdir}

# for -devel
install -m 0644 -p ./caffe2/core/*.h  %{buildroot}%{_includedir}/caffe2/core/
install -m 0644 -p ./caffe2/proto/*.h  %{buildroot}%{_includedir}/caffe2/proto/

# to convert caffe to caffe2 models
install -m 0644 -p ./caffe/proto/caffe.pb.h  %{buildroot}%{_includedir}/caffe/proto/

# install python packages
cp -rf ./caffe2/python/* %{buildroot}%{python_sitelib}/caffe2/

popd

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-,root,root)
%manifest %{name}.manifest
%doc ./LICENSE
%doc ./PATENTS
%{_bindir}/*
%{_libdir}/libCaffe2_CPU.so

%files devel
%{_includedir}/caffe2/*
%{_includedir}/caffe/proto/*

%files python
%{python_sitelib}/caffe2/*

%changelog
* Thu Dec 28 2017 Geunsik Lim <geunsik.lim@samsung.com>
- Tizen 4.0 (snapshot), Caffe2 0.7.0 f71695d (on Apr-19-2017)

