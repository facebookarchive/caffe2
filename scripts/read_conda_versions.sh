# Simple script used to easily search all packages in conda for their
# dependency requirements

if [ -z "$CONDA_ROOT" ]; then
  echo "Please set CONDA_ROOT so that I know where to search for conda libraries"
  exit 1
fi

if [ -z "$1" ]; then
  echo "Please give me a package name to search for"
  exit 1
fi
PKG_NAME="$1"

if [ -n "$2" ]; then
  echo "Searching in channel $2"
  CONDA_CHANNEL="$2"
fi

# These are the packages of interest to search the dependencies for
# TODO use this
PACKAGES_OF_INTEREST=( libgcc-ng libprotobuf numpy )

# We only care about shared/dynamic libraries
if [[ "$(uname)" == 'Darwin' ]]; then
  LIB_SUFFIX='.dylib'
else
  LIB_SUFFIX='.so'
fi

# Split the output from conda search into an array, one line per package (plus
# the header)
conda_search_packages=()
while read -r line; do conda_search_packages+=("$line"); done <<< "$(conda search $PKG_NAME $CONDA_CHANNEL)"

### Typical `conda search` output looks like
###   Loading channels: done
###   Name                       Version                   Build  Channel
###   protobuf                   2.6.1                    py27_0  defaults
###                              2.6.1                    py27_1  defaults
###                              3.2.0                    py27_0  defaults
###                              3.2.0                    py35_0  defaults
###                              3.2.0                    py36_0  defaults
###                              3.4.1            py27h66c1d77_0  defaults
###                              3.4.1            py35h9d33684_0  defaults
###                              3.4.1            py36h314970b_0  defaults
###                              3.5.1            py27h0a44026_0  defaults
###                              3.5.1            py35h0a44026_0  defaults
###                              3.5.1            py36h0a44026_0  defaults
##
### Typical `conda info` output looks like
###   protobuf 3.5.1 py36h0a44026_0 #     -----------------------------
###   file name   : protobuf-3.5.1-py36h0a44026_0.tar.bz2
###   name        : protobuf
###   version     : 3.5.1
###   build string: py36h0a44026_0
###   build number: 0
###   channel     : https://repo.continuum.io/pkgs/main/osx-64
###   size        : 589 KB
###   arch        : None
###   constrains  : ()
###   license     : New BSD License
###   license_family: BSD
###   md5         : 7dbdb06612e21c42fbb8a62354e13e10
###   platform    : None
###   subdir      : osx-64
###   timestamp   : 1519951502766
###   url         : https://repo.continuum.io/pkgs/main/osx-64/protobuf-3.5.1-py36h0a44026_0.tar.bz2
###   dependencies:
###       libcxx >=4.0.1
###       libprotobuf >=3.5.1,<3.6.0a0
###       python >=3.6,<3.7.0a0
###       six

# Echo what packages we'll look through.
echo "Processing these packages:"
for pkg in "${conda_search_packages[@]:2}"; do
  echo "  $pkg"
done


# Redirect all `conda install` and `conda uninstall` output to a separate file
# cause it's so noisy
CONDA_INSTALL_LOG='OUT_read_conda_versions.txt'
echo "Conda install/uninstall log for $PKG_NAME" > $CONDA_INSTALL_LOG

# Look up each package in conda info
for pkg in "${conda_search_packages[@]:2}"; do

  # Split each line into an array and build the package specification
  # <package_name on 1st line only>  maj.min.patch  build_string  channel_name
  line_parts=( $pkg )
  if [[ ${line_parts[0]} == $PKG_NAME ]]; then
    # First line of output
    PKG_VERSION="${line_parts[1]}"
    PKG_BUILD_STR="${line_parts[2]}"
  else
    PKG_VERSION="${line_parts[0]}"
    PKG_BUILD_STR="${line_parts[1]}"
  fi
  PKG_SPEC="$PKG_NAME=$PKG_VERSION=$PKG_BUILD_STR"

  # Output current pkg spec
  echo
  echo "${PKG_SPEC}:"

  # Split the output of conda_info into an array of lines
  pkg_dependencies=()
  while read -r line; do pkg_dependencies+=("$line"); done <<< "$(conda info "$PKG_SPEC" $CONDA_CHANNEL)"

  # List all the listed dependencies in `conda info`
  echo "  Listed dependencies:"
  for pkg_dependency in "${pkg_dependencies[@]:20}"; do
    echo "    $PKG_SPEC is $pkg_dependency"
  done

  # But sometimes (a lot of the time) the gcc with which a package was built
  # against is not listed in dependencies. So we try to figure it out manually
  # We install this exact package, and then grep the exported symbols for signs
  # of cxx11
  conda uninstall -y "$PKG_NAME" --quiet > $CONDA_INSTALL_LOG 2>&1
  conda install -y "$PKG_SPEC" --quiet $CONDA_CHANNEL > $CONDA_INSTALL_LOG 2>&1
  if [ $? -eq 0 ]; then

    MENTIONS_CXX11="$(nm "$CONDA_ROOT/lib/lib${PKG_NAME}${LIB_SUFFIX}" | grep cxx11 | wc -l)"
    if [ $MENTIONS_CXX11 -gt 0 ]; then
      echo "  This package is built against the recent gcc ABI ($MENTIONS_CXX11 mentions of cxx11)"
    fi
  else
    echo "Error installing $PKG_SPEC , continuing"
  fi
done
