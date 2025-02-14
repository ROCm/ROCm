#!/bin/bash

# compute_helper.sh is created to add common functions similar to compute_utils.sh in stg1
# stg2 build scripts can make use of common function
# TODO:  Component build scripts can be optimized by adding common function in this file(ex for ASAN, Debugging  etc)
#        All build scripts should use the common function.

set -e
set -o pipefail
# Set the LLVM directory path with respect to ROCM_PATH
# LLVM is installed in $ROCM_PATH/lib/llvm
ROCM_LLVMDIR="lib/llvm"
# Set ADDRESS_SANITIZER to OFF by default
export ADDRESS_SANITIZER="OFF"

DISABLE_DEBUG_PACKAGE="false"
export CFLAGS="$CFLAGS $SET_DWARF_VERSION_4  "
export CXXFLAGS="$CXXFLAGS $SET_DWARF_VERSION_4 "

# Print message to stderr
#   param message string to print on exit
# Example: printErr "file not found"
printErr() {
    echo "$@" 1>&2
}

# Print message to stderr and terminate current program
#   param message string to print on exit
# Example: die "Your program" has terminated
die() {
    printErr "FATAL: $@"
    exit 1
}

# Die if first argument is empty
#   param string to validate
#   param error message
# Example: die "$VARIABLE" "Your program" has terminated
dieIfEmpty() {
    if [ "$1" == "" ]; then
        shift
        die "$@"
    fi
}

ack_and_skip_static() {
    echo "static build is not enabled for this component..skipping"
    exit 0
}

# Copy a file or directory to target location and show single line progress
progressCopy() {
    if [ -d "$1" ]; then
        rsync -a "$1"/* "$2"
    else
        rsync -a "$1" "$2"
    fi
}

# Get OS identification string
# xargs will remove the trailing whitespaces
getOsVersion() {
    lsb_release -d | cut -d: -f2 | xargs
}

# Get kernel identification string
getKernelVersion() {
    uname -r
}

# Trim excessive whitespace from a string
strTrim() {
    echo "$@" | xargs
}

# Return whether the booted OS is Fedora or not
isFedora() {
    grep -iq fedora /etc/os-release
}

# Return whether the booted system is EFI or not
isEFI() {
    [ -d "/sys/firmware/efi" ]
}

print_lib_type() {
   if [ "$1" == "OFF" ];
   then
       echo " Building Archive "
   else
       echo " Building Shared Object "
   fi
}

# Get CMAKE build flags for CMAKE build trigger
set_build_variables() {
    local cmake_cxx_flag_params
    local cmake_cc_flag_params
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ] ; then
        cmake_cxx_flag_params="$ROCM_PATH/llvm/bin/clang++"
        cmake_cc_flag_params="$ROCM_PATH/llvm/bin/clang"
        cmake_comp_params=(
            "-DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang"
            "-DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++"
        )
    else
        cmake_cxx_flag_params="$ROCM_PATH/bin/hipcc"
        cmake_cc_flag_params="$ROCM_PATH/bin/hipcc"
        cmake_comp_params=(
            "-DCMAKE_C_COMPILER=$ROCM_PATH/bin/hipcc"
            "-DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc"
        )
    fi

    case "$1" in
    ("CXX")
      printf "%s" "${cmake_cxx_flag_params}"
      ;;
     ("CC")
      printf "%s" "${cmake_cc_flag_params}"
      ;;
     ("CMAKE_C_CXX")
      printf '%s ' "${cmake_comp_params[@]}"
      ;;
     (*)
      exit 1
      ;;
    esac
    exit
}

# Get the directory containing the cmake config files
getCmakePath() {
    local rocmInstallPath=${ROCM_PATH}
    local cmakePath="lib/cmake"
    dieIfEmpty "$rocmInstallPath"
    echo "$rocmInstallPath/$cmakePath"
}

getInstallLibDir() {
    local libDir="lib"
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ] ; then
        libDir="lib/asan"
    fi
    echo "$libDir"
}

set_asan_env_vars() {
    # Flag to set cmake build params for ASAN builds
    ASAN_CMAKE_PARAMS="true"
    export ADDRESS_SANITIZER="ON"
    LLVM_BIN_DIR="${ROCM_PATH}/llvm/bin"
    export CC="$LLVM_BIN_DIR/clang"
    export CXX="$LLVM_BIN_DIR/clang++"
    export FC="$LLVM_BIN_DIR/flang"
    export PATH="$LLVM_BIN_DIR/:$PATH"
    # get exact path to ASAN lib containing clang version
    ASAN_LIB_PATH=$(clang --print-file-name=libclang_rt.asan-x86_64.so)
    export LD_LIBRARY_PATH="${ASAN_LIB_PATH%/*}:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ASAN_OPTIONS="detect_leaks=0"
}

set_address_sanitizer_on() {
    export CFLAGS="-fsanitize=address -shared-libasan $SET_DWARF_VERSION_4 -g -gz"
    export CXXFLAGS="-fsanitize=address -shared-libasan $SET_DWARF_VERSION_4 -g -gz"
    export LDFLAGS="-Wl,--enable-new-dtags -fuse-ld=lld  -fsanitize=address -shared-libasan -g -gz -Wl,--build-id=sha1 -L${ROCM_PATH}/llvm/lib -L${ROCM_PATH}/lib/asan"
}

rebuild_lapack() {
    wget -nv -O lapack-3.9.1.tar.gz \
        http://compute-artifactory.amd.com/artifactory/rocm-generic-thirdparty-deps/ubuntu/lapack-v3.9.1.tar.gz
    sh -c "echo 'd0085d2caf997ff39299c05d4bacb6f3d27001d25a4cc613d48c1f352b73e7e0 *lapack-3.9.1.tar.gz' | sha256sum -c"
    tar xzf lapack-3.9.1.tar.gz --one-top-level=lapack-src --strip-components 1
    rm lapack-3.9.1.tar.gz

    cmake -Slapack-src -Blapack-bld \
        ${LAUNCHER_FLAGS} \
        -DBUILD_TESTING=OFF \
        -DCBLAS=ON \
        -DLAPACKE=OFF
    cmake --build lapack-bld -- -j${PROC}
    cmake --build lapack-bld -- install
    rm -r lapack-src lapack-bld
}

# debug function #dumping values in case of error to solve the same
print_vars() {
echo " Status of Vars in $1 build "
echo " TARGET= $2 "
echo " BUILD_TYPE = $3 "
echo " SHARED_LIBS = $4 "
echo " CLEAN_OR_OUT = $5 "
echo " PKGTYPE= $6 "
echo " MAKETARGET = $7 "
}

rocm_math_common_cmake_params=()
init_rocm_common_cmake_params(){
  local retCmakeParams=${1:-rocm_math_common_cmake_params}
  local SET_BUILD_TYPE=${BUILD_TYPE:-'RelWithDebInfo'}
  local ASAN_LIBDIR="lib/asan"
  local CMAKE_PATH=$(getCmakePath)
# Common cmake parameters can be set
# component build scripts can use this function
  local cmake_params
  if [ "${ASAN_CMAKE_PARAMS}" == "true" ] ; then
    cmake_params=(
        "-DCMAKE_PREFIX_PATH=$CMAKE_PATH;${ROCM_PATH}/$ASAN_LIBDIR;$ROCM_PATH/llvm;$ROCM_PATH"
        "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_ASAN_LIB_RPATH"
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_ASAN_EXE_RPATH"
        "-DENABLE_ASAN_PACKAGING=true"
    )
  else
    cmake_params=(
        "-DCMAKE_PREFIX_PATH=${ROCM_PATH}/llvm;${ROCM_PATH}"
        "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LIB_RPATH"
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_EXE_RPATH"
    )
  fi

  cmake_params+=(
      "-DCMAKE_VERBOSE_MAKEFILE=1"
      "-DCMAKE_BUILD_TYPE=${SET_BUILD_TYPE}"
      "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"
      "-DCMAKE_INSTALL_PREFIX=${ROCM_PATH}"
      "-DCMAKE_PACKAGING_INSTALL_PREFIX=${ROCM_PATH}"
      "-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF"
      "-DROCM_SYMLINK_LIBS=OFF"
      "-DCPACK_PACKAGING_INSTALL_PREFIX=${ROCM_PATH}"
      "-DROCM_DISABLE_LDCONFIG=ON"
      "-DCPACK_SET_DESTDIR=OFF"
      "-DCPACK_RPM_PACKAGE_RELOCATABLE=ON"
      "-DROCM_PATH=${ROCM_PATH}"
  )

  #TODO :remove if clause once debug related issues are fixed
  if [ "${DISABLE_DEBUG_PACKAGE}" == "true" ] ; then
    SET_BUILD_TYPE=${BUILD_TYPE:-'Release'}
    cmake_params+=(
        "-DCPACK_DEBIAN_DEBUGINFO_PACKAGE=FALSE"
        "-DCPACK_RPM_DEBUGINFO_PACKAGE=FALSE"
        "-DCPACK_RPM_INSTALL_WITH_EXEC=FALSE"
        "-DCMAKE_BUILD_TYPE=${SET_BUILD_TYPE}"
    )
  elif [ "$SET_BUILD_TYPE" == "RelWithDebInfo" ] || [ "$SET_BUILD_TYPE" == "Debug" ]; then
    # RelWithDebinfo optimization level -O2 is having performance impact
    # So overriding the same to -O3
    cmake_params+=(
        "-DCPACK_DEBIAN_DEBUGINFO_PACKAGE=TRUE"
        "-DCPACK_RPM_DEBUGINFO_PACKAGE=TRUE"
        "-DCPACK_RPM_INSTALL_WITH_EXEC=TRUE"
        "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3 -g -DNDEBUG"
    )
  fi
    eval "${retCmakeParams}=( \"\${cmake_params[@]}\" ) "
}

disable_debug_package_generation(){
    case "$DISTRO_ID" in
       (sles*|rhel*)
           export CFLAGS=${CFLAGS//"$SET_DWARF_VERSION_4"}
           export CXXFLAGS=${CXXFLAGS//"$SET_DWARF_VERSION_4"}
           ;;
      (*)
           ;;
    esac
    DISABLE_DEBUG_PACKAGE="true"
}

# Setup a number of variables to specify where to find the source
# where to do the build and where to put the packages
# Note the PACKAGE_DIR downcases the package name
# This could be extended to do different things based on $1
set_component_src(){
    COMPONENT_SRC="$LIBS_WORK_DIR/$1"

    BUILD_DIR="$OUT_DIR/build/$1"
    DEB_PATH="$OUT_DIR/${PKGTYPE}/${1,,}"
    RPM_PATH="$OUT_DIR/${PKGTYPE}/${1,,}"
    PACKAGE_DIR="$OUT_DIR/${PKGTYPE}/${1,,}"
}

# Standard definition of function to print the package location.  If
# for some reason a custom version is needed then it can overwrite
# this definition
# TODO: Don't use a global PKGTYPE, pass the value in as a parameter
print_output_directory() {
    case ${1:-$PKGTYPE} in
        ("deb")
            echo ${DEB_PATH};;
        ("rpm")
            echo ${RPM_PATH};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

# Standard argument processing
# Here to avoid repetition
TARGET="build"			# default target
stage2_command_args(){
    while [ "$1" != "" ];
    do
	case $1 in
            -o  | --outdir )
		shift 1; PKGTYPE=$1 ; TARGET="outdir" ;;
            -c  | --clean )
		TARGET="clean" ;;
            *)
		break ;;
	esac
	shift 1
    done
}

show_build_cache_stats(){
    if [ "$CCACHE_ENABLED" = "true" ] ; then
	if ! ccache -s ; then
	    echo "Unable to display ccache stats"
	fi
    fi
}
