# ---[ OpenCL
if(USE_OPENCL)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_SOURCE_DIR}/third_party/libopencl-stub/src/libopencl.c)

  # ---[ CPU files.
  file(GLOB tmp *.cc)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
  file(GLOB tmp *_test.cc)
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})

  add_subdirectory(operators)

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
endif()
