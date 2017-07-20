# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

# Locate an OpenCL implementation.
# Currently supports AMD APP SDK (http://developer.amd.com/sdks/AMDAPPSDK/Pages/default.aspx/)
#
# Defines the following variables:
#
#   OPENCL_FOUND - Found the OPENCL framework
#   OPENCL_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   OPENCL_LIBRARIES - libopencl
#
# Accepts the following variables as input:
#
#   OPENCL_ROOT - (as a CMake or environment variable)
#                The root directory of the OpenCL implementation found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findOpenCL should search for
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(OPENCL REQUIRED)
#    include_directories(${OPENCL_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${OPENCL_LIBRARIES})
#
#-----------------------
include( CheckSymbolExists )
include( CMakePushCheckState )

if( DEFINED OPENCL_ROOT OR DEFINED ENV{OPENCL_ROOT})
  message( STATUS "Defined OPENCL_ROOT: ${OPENCL_ROOT}, ENV{OPENCL_ROOT}: $ENV{OPENCL_ROOT}" )
endif( )

find_path(OPENCL_INCLUDE_DIRS
  NAMES OpenCL/cl.h CL/cl.h
  HINTS
    ${OPENCL_ROOT}/include
    $ENV{OPENCL_ROOT}/include
    $ENV{AMDAPPSDKROOT}/include
    $ENV{CUDA_PATH}/include
  PATHS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
  DOC "OpenCL header file path"
)
mark_as_advanced( OPENCL_INCLUDE_DIRS )
message( STATUS "OPENCL_INCLUDE_DIRS: ${OPENCL_INCLUDE_DIRS}" )

set( OpenCL_VERSION "0.0" )

cmake_push_check_state( RESET )
set( CMAKE_REQUIRED_INCLUDES "${OPENCL_INCLUDE_DIRS}" )

# Bug in check_symbol_exists prevents us from specifying a list of files, so we loop
# Only 1 of these files will exist on a system, so the other file will not clobber the output variable
if( APPLE )
   set( CL_HEADER_FILE "OpenCL/cl.h" )
else( )
   set( CL_HEADER_FILE "CL/cl.h" )
endif( )

check_symbol_exists( CL_VERSION_2_0 ${CL_HEADER_FILE} HAVE_CL_2_0 )
check_symbol_exists( CL_VERSION_1_2 ${CL_HEADER_FILE} HAVE_CL_1_2 )
check_symbol_exists( CL_VERSION_1_1 ${CL_HEADER_FILE} HAVE_CL_1_1 )
# message( STATUS "HAVE_CL_2_0: ${HAVE_CL_2_0}" )
# message( STATUS "HAVE_CL_1_2: ${HAVE_CL_1_2}" )
# message( STATUS "HAVE_CL_1_1: ${HAVE_CL_1_1}" )

# set OpenCL_VERSION to the highest detected version
if( HAVE_CL_2_0 )
  set( OpenCL_VERSION "2.0" )
elseif( HAVE_CL_1_2 )
  set( OpenCL_VERSION "1.2" )
elseif( HAVE_CL_1_1 )
  set( OpenCL_VERSION "1.1" )
endif( )

cmake_pop_check_state( )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
if( LIB64 )
  message( STATUS "FindOpenCL searching for 64-bit libraries" )
else( )
  message( STATUS "FindOpenCL searching for 32-bit libraries" )
endif( )

if( LIB64 )
  find_library( OPENCL_LIBRARIES
    NAMES OpenCL
    HINTS
      ${OPENCL_ROOT}/lib
      $ENV{OPENCL_ROOT}/lib
      $ENV{AMDAPPSDKROOT}/lib
      $ENV{CUDA_PATH}/lib
    DOC "OpenCL dynamic library path"
    PATH_SUFFIXES x86_64 x64 x86_64/sdk
    PATHS
    /usr/lib
    /usr/local/cuda/lib
  )
else( )
  find_library( OPENCL_LIBRARIES
    NAMES OpenCL
    HINTS
      ${OPENCL_ROOT}/lib
      $ENV{OPENCL_ROOT}/lib
      $ENV{AMDAPPSDKROOT}/lib
      $ENV{CUDA_PATH}/lib
    DOC "OpenCL dynamic library path"
    PATH_SUFFIXES x86 Win32
    PATHS
    /usr/lib
    /usr/local/cuda/lib
  )
endif( )
mark_as_advanced( OPENCL_LIBRARIES )

# message( STATUS "OpenCL_FIND_VERSION: ${OpenCL_FIND_VERSION}" )
if( OpenCL_VERSION VERSION_LESS OpenCL_FIND_VERSION )
    message( FATAL_ERROR "Requested OpenCL version: ${OpenCL_FIND_VERSION}, Found OpenCL version: ${OpenCL_VERSION}" )
endif( )

# If we asked for OpenCL 1.2, and we found a version installed greater than that, pass the 'use deprecated' flag
if( (OpenCL_FIND_VERSION VERSION_LESS "2.0") AND (OpenCL_VERSION VERSION_GREATER OpenCL_FIND_VERSION) )
    add_definitions( -DCL_USE_DEPRECATED_OPENCL_2_0_APIS )

    # If we asked for OpenCL 1.1, and we found a version installed greater than that, pass the 'use deprecated' flag
    if( (OpenCL_FIND_VERSION VERSION_LESS "1.2") AND (OpenCL_VERSION VERSION_GREATER OpenCL_FIND_VERSION) )
        add_definitions( -DCL_USE_DEPRECATED_OPENCL_1_1_APIS )
    endif( )
endif( )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( OPENCL
    REQUIRED_VARS OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS
    VERSION_VAR OpenCL_VERSION
    )

if( NOT OPENCL_FOUND )
    message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
else( )
    message(STATUS "FindOpenCL ${OPENCL_LIBRARIES}, ${OPENCL_INCLUDE_DIRS}")
endif()
