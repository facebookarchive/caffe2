# ---[ snappy
# Find the Snappy libraries
#
# The following variables are optionally searched for defaults
#  SNAPPY_ROOT_DIR:    Base directory where all Snappy components are found
#
# The following are set after configuration is done:
#  SNAPPY_FOUND
#  Snappy_INCLUDE_DIR
#  Snappy_LIBRARIES

# Use pkg-config to help find libraries if it's available
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(Snappy Snappy)
endif()

# Look for the header files.
find_path(Snappy_INCLUDE_DIR
    NAMES snappy.h
    PATHS ${SNAPPY_ROOT_DIR}/include ${Snappy_INCLUDE_DIRS}
)

# Look for the library.
find_library(Snappy_LIBRARY
    NAMES snappy
    PATHS ${SNAPPY_ROOT_DIR}/lib ${Snappy_LIBRARY_DIRS}
)

# Now delegate to this cmake function to make sure all paths exist
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Snappy
    FOUND_VAR SNAPPY_FOUND
    REQUIRED_VARS Snappy_LIBRARY Snappy_INCLUDE_DIR
)

if(NOT SNAPPY_FOUND)
  return()
endif()

message(STATUS "Found Snappy  (include: ${Snappy_INCLUDE_DIR}, library: ${Snappy_LIBRARY})")
mark_as_advanced(Snappy_INCLUDE_DIR Snappy_LIBRARY)

# Create a Snappy target
if (NOT TARGET snappy::snappy)
  add_library(snappy::snappy UNKNOWN IMPORTED)
endif()

# Set target properties for the new snappy target
set_target_properties(snappy::snappy PROPERTIES
  IMPORTED_LOCATION "${Snappy_LIBRARY}"
  INTERFACE_COMPILE_OPTIONS "${Snappy_CFLAGS_OTHER}"
  INTERFACE_INCLUDE_DIRECTORIES "${Snappy_INCLUDE_DIR}"
)

# Read the Snappy version from the header
caffe_parse_header(${Snappy_INCLUDE_DIR}/snappy-stubs-public.h
                     SNAPPY_VERION_LINES SNAPPY_MAJOR SNAPPY_MINOR SNAPPY_PATCHLEVEL)
set(Snappy_VERSION "${SNAPPY_MAJOR}.${SNAPPY_MINOR}.${SNAPPY_PATCHLEVEL}")

