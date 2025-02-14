#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
    type referred to by pkg_type"
    echo "  -h,  --help             Prints this help"
    echo "  -r,  --release          Make a release build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -s,  --static             Supports static CI by accepting this param & not bailing out. No effect of the param though"
    echo

    return 0
}

TARGET="build"
MAKEOPTS="$DASH_JAY"
HIPIFY_CLANG_BUILD_DIR="$(getBuildPath $HIPIFY_ROOT)"
BUILD_TYPE="Debug"
PACKAGE_ROOT="$(getPackageRoot)"
HIPIFY_CLANG_HASH=""
LIGHTNING_PATH="$ROCM_INSTALL_PATH/llvm"
ADDRESS_SANITIZER=false
INSTALL_CLANG_HEADERS="OFF"
DEB_PATH="$(getDebPath hipify)"
RPM_PATH="$(getRpmPath hipify)"
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
                set_address_sanitizer_on
                ADDRESS_SANITIZER=true ; shift ;;
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


clean_hipify() {
    echo "Cleaning hipify-clang"
    rm -rf "$HIPIFY_CLANG_BUILD_DIR"
    rm -rf "$DEB_PATH"
    rm -rf "$RPM_PATH"
}

package_hipify() {
    if [ "$PACKAGEEXT" = "deb" ]; then
        rm -rf "$DEB_PATH"
        mkdir -p "$DEB_PATH"
    fi

    if [ "$PACKAGEEXT" = "rpm" ]; then
        rm -rf "$RPM_PATH"
        mkdir -p "$RPM_PATH"
    fi

    pushd "$HIPIFY_CLANG_BUILD_DIR"
    make $MAKEOPTS package_hipify-clang
    popd

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$DEB_PATH"  $HIPIFY_CLANG_BUILD_DIR/hipify*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$RPM_PATH"  $HIPIFY_CLANG_BUILD_DIR/hipify*.rpm
}

build_hipify() {
    echo "Building hipify-clang binaries"
    mkdir -p "$HIPIFY_CLANG_BUILD_DIR"

    pushd "$HIPIFY_CLANG_BUILD_DIR"
    cmake \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        $(rocm_common_cmake_params) \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
        -DCPACK_PACKAGING_INSTALL_PREFIX=$ROCM_INSTALL_PATH \
        -DCMAKE_PREFIX_PATH="$LIGHTNING_PATH" \
        -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
        -DHIPIFY_INSTALL_CLANG_HEADERS="$INSTALL_CLANG_HEADERS" \
        $HIPIFY_ROOT

    cmake --build . -- $MAKEOPTS install
    popd
    pushd "$HIPIFY_ROOT"
        HIPIFY_CLANG_HASH=`git describe --dirty --long --match [0-9]* --always`
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
    (clean) clean_hipify ;;
    (build) build_hipify; package_hipify ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
