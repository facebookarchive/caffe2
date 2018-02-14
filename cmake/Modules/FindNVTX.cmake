# - Try to find CUDA NVTX
#
# The following variables are optionally searched for defaults
#  NVTX_ROOT_DIR:            Base directory where all NVTX components are found
#
# The following are set after configuration is done:
#  NVTX_FOUND
#  NVTX_INCLUDE_DIRS
#  NVTX_LIBRARIES

include(FindPackageHandleStandardArgs)

set(NVTX_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NVTX")

find_path(NVTX_INCLUDE_DIR nvToolsExt.h
    HINTS ${NVTX_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

find_library(NVTX_LIBRARY nvToolsExt
    HINTS ${NVTX_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_package_handle_standard_args(
    NVTX DEFAULT_MSG NVTX_INCLUDE_DIR NVTX_LIBRARY)

if(NVTX_FOUND)
  set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
  set(NVTX_LIBRARIES ${NVTX_LIBRARY})
  message(STATUS "Found NVTX: (include: ${NVTX_INCLUDE_DIR}, library: ${NVTX_LIBRARY})")
  mark_as_advanced(NVTX_ROOT_DIR NVTX_LIBRARY NVTX_INCLUDE_DIR)
endif()
