#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...] [make options]"
    echo
    echo "Options:"
    echo "  -h,  --help               Prints this help"
    echo "  -s,  --static             Supports static CI by accepting this param & not bailing out. No effect of the param though"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of type referred to by pkg_type"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}
PROJ_NAME="opencl-on-rocclr"
MAKEOPTS="$DASH_JAY"

BUILD_PATH="$(getBuildPath $PROJ_NAME)"

TARGET="build"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_DEB="$PACKAGE_ROOT/deb/$PROJ_NAME"
PACKAGE_RPM="$PACKAGE_ROOT/rpm/$PROJ_NAME"
CORE_BUILD_DIR="$(getBuildPath hsa-core)"
ROCclr_BUILD_DIR="$(getBuildPath rocclr)"
BUILD_TYPE="Debug"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"


VALID_STR=`getopt -o hcraso: --long help,clean,release,static,address_sanitizer,outdir: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                ack_and_skip_static ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        --)     shift; break;;
        (*)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac

done

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi

clean_opencl_on_rocclr() {
    rm -rf "$BUILD_PATH"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$PACKAGE_ROOT/bin/clinfo"
    rm -rf "$PACKAGE_ROOT/lib/libOpenCL.so*"
    rm -rf "$PACKAGE_ROOT/lib/libamdocl64.so"
    rm -rf "$PACKAGE_ROOT/lib/libcltrace.so"
}

build_opencl_on_rocclr() {
    if [  -e "$CLR_ROOT/CMakeLists.txt" ]; then
        _OCL_CMAKELIST_DIR="$CLR_ROOT"
        _OCL_CMAKELIST_OPT="-DCLR_BUILD_HIP=OFF -DCLR_BUILD_OCL=ON"
    elif [ ! -e "$OPENCL_ON_ROCclr_ROOT/CMakeLists.txt" ]; then
        echo "No $OPENCL_ON_ROCclr_ROOT/CMakeLists.txt file, skipping opencl on rocclr" >&2
        echo "No $OPENCL_ON_ROCclr_ROOT/CMakeLists.txt file, skipping opencl on rocclr"
        exit 0
    else
         _OCL_CMAKELIST_DIR="$OPENCL_ON_ROCclr_ROOT"
         _OCL_CMAKELIST_OPT=""
    fi

    echo "$_OCL_CMAKELIST_DIR"
    mkdir -p "$BUILD_PATH"
    pushd "$BUILD_PATH"

    if [ ! -e Makefile ]; then
        echo "Building OpenCL-On-ROCclr CMake environment"

        cmake \
            $(rocm_cmake_params) \
            -DUSE_COMGR_LIBRARY=ON \
            $(rocm_common_cmake_params) \
            -DLINK_COMGR_STATIC="no" \
            -DBUILD_TESTS=ON \
            $_OCL_CMAKELIST_OPT \
            "$_OCL_CMAKELIST_DIR"

        echo "CMake complete"
    fi

    echo "Building OpenCL-On-ROCclr"
    cmake --build . -- $MAKEOPTS

    echo "Installing OpenCL-On-ROCclr"
    cmake --build . -- $MAKEOPTS install

    popd
}

package_opencl_on_rocclr() {
    echo "Packaging OpenCL-On-ROCclr"
    pushd "$BUILD_PATH"
    cmake --build . -- package
    mkdir -p $PACKAGE_DEB
    mkdir -p $PACKAGE_RPM
    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" *.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" *.rpm
    popd
}

print_output_directory() {
     case ${PKGTYPE} in
         ("deb")
             echo ${PACKAGE_DEB};;
         ("rpm")
             echo ${PACKAGE_RPM};;
         (*)
             echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
     esac
     exit
}

case $TARGET in
    (clean) clean_opencl_on_rocclr ;;
    (build) build_opencl_on_rocclr ; package_opencl_on_rocclr ;;
    (outdir) print_output_directory ;;
        (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
