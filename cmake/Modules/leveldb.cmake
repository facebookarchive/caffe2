# ---[ LevelDB
# - Find LevelDB
#
#  LevelDB_INCLUDES  - List of LevelDB includes
#  LevelDB_LIBRARIES - List of libraries when using LevelDB.
#  LevelDB_FOUND     - True if LevelDB found.

# Use pkg-config to help find libraries if it's available
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(LevelDB LevelDB)
endif()

# Look for the header files.
find_path(LevelDB_INCLUDE_DIR
    NAMES leveldb/db.h
    PATHS $ENV{LEVELDB_ROOT}/include ${LevelDB_INCLUDE_DIRS} /opt/local/include /usr/local/include /usr/include
    DOC "Path in which the file leveldb/db.h is located."
)

# Look for the library.
find_library(LevelDB_LIBRARY
    NAMES leveldb
    PATHS $ENV{LEVELDB_ROOT}/lib ${LevelDB_LIBRARY_DIRS} /opt/local/lib /usr/local/lib /usr/lib
    DOC "Path to leveldb library."
)

# Now delegate to this cmake function to make sure all paths exist
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LevelDB
    FOUND_VAR LevelDB_FOUND
    REQUIRED_VARS LevelDB_LIBRARY LevelDB_INCLUDE_DIR
)

if (NOT LevelDB_FOUND)
  return()
endif()

message(STATUS "Found LevelDB (include: ${LevelDB_INCLUDE}, library: ${LevelDB_LIBRARY})")
mark_as_advanced(LevelDB_INCLUDE_DIR LevelDB_LIBRARY)

# Create a leveldb target
if (NOT TARGET leveldb::leveldb)
  add_library(leveldb::leveldb UNKNOWN IMPORTED)
endif()

# Set target properties for the new leveldb target
set_target_properties(leveldb::leveldb PROPERTIES
  IMPORTED_LOCATION "${LevelDB_LIBRARY}"
  INTERFACE_COMPILE_OPTIONS "${LevelDB_CFLAGS_OTHER}"
  INTERFACE_INCLUDE_DIRECTORIES "${LevelDB_INCLUDE_DIR}"
)

# Read the LevelDB version from the header
if(EXISTS "${LevelDB_INCLUDE_DIR}/leveldb/db.h")
  file(STRINGS "${LevelDB_INCLUDE_DIR}/leveldb/db.h" __version_lines
         REGEX "static const int k[^V]+Version[ \t]+=[ \t]+[0-9]+;")

  foreach(__line ${__version_lines})
    if(__line MATCHES "[^k]+kMajorVersion[ \t]+=[ \t]+([0-9]+);")
      set(LEVELDB_VERSION_MAJOR ${CMAKE_MATCH_1})
    elseif(__line MATCHES "[^k]+kMinorVersion[ \t]+=[ \t]+([0-9]+);")
      set(LEVELDB_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
  endforeach()

  if(LEVELDB_VERSION_MAJOR AND LEVELDB_VERSION_MINOR)
    set(LEVELDB_VERSION "${LEVELDB_VERSION_MAJOR}.${LEVELDB_VERSION_MINOR}")
  endif()

  # caffe_clear_vars(__line __version_lines)
endif()
