#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -r,  --release            Build a release version of the package"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -s,  --static             Supports static CI by accepting this param & not bailing out. No effect of the param though"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
    type referred to by pkg_type"
    echo "  -h,  --help             Prints this help"
    echo
    echo

    return 0
}
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_BIN="$(getBinPath)"
PACKAGE_LIB="$(getLibPath)"
BUILD_PATH="$(getBuildPath devicelibs)"
INSTALL_PATH="$(getPackageRoot)"
LIGHTNING_BUILD_PATH="$(getBuildPath lightning)"
DEB_PATH="$(getDebPath devicelibs)"
RPM_PATH="$(getRpmPath devicelibs)"
AMDGCN_LIB_PATH="$PACKAGE_ROOT/amdgcn/bitcode/"

TARGET="build"
MAKEOPTS="$DASH_JAY"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
PKGTYPE="deb"
MAKETARGET="deb"

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
                BUILD_TYPE="Release" ; shift ;;
        (-a | --address_sanitizer)
                ASAN_CMAKE_PARAMS="true"
                ack_and_ignore_asan ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        --)     shift; break;; # end delimiter
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


clean_devicelibs() {
    rm -rf "$BUILD_PATH"
    rm -f $PACKAGE_LIB/hc*.bc
    rm -f $PACKAGE_LIB/irif*.bc
    rm -f $PACKAGE_LIB/ockl*.bc
    rm -f $PACKAGE_LIB/oclc*.bc
    rm -f $PACKAGE_LIB/ocml*.bc
    rm -f $PACKAGE_LIB/opencl*.bc
    rm -f $PACKAGE_LIB/openmp*.bc
    rm -rf $PACKAGE_ROOT/amdgcn
    rm -rf "$DEB_PATH"
    rm -rf "$RPM_PATH"
}

build_devicelibs() {
    mkdir -p "$BUILD_PATH"
    pushd "$BUILD_PATH"

    local clangResourceDir="$($LIGHTNING_BUILD_PATH/bin/clang -print-resource-dir)"
    local clangResourceVer=${clangResourceDir#*lib/clang/}
    local bitcodeInstallLoc="lib/llvm/lib/clang/${clangResourceVer}/lib"

    export LLVM_BUILD="$LIGHTNING_BUILD_PATH"
    if [ ! -e Makefile ]; then
        cmake $(rocm_cmake_params) \
              $(rocm_common_cmake_params) \
              -DBUILD_SHARED_LIBS=$SHARED_LIBS \
              -DROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW="$bitcodeInstallLoc/amdgcn" \
              -DROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD="amdgcn" \
              "$DEVICELIBS_ROOT"

        echo "CMake complete"
    fi

    echo "Building device-libs"
    cmake --build . -- $MAKEOPTS
    cmake --build . -- $MAKEOPTS install
    popd
}

package_devicelibs() {
    mkdir -p "$DEB_PATH"
    mkdir -p "$RPM_PATH"
    pushd "$BUILD_PATH"
    cpack
    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$DEB_PATH" *.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$RPM_PATH" *.rpm 
    popd
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${DEB_PATH};;
        ("rpm")
            echo ${RPM_PATH};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}
case $TARGET in
    (clean) clean_devicelibs ;;
    (build) build_devicelibs; package_devicelibs ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
