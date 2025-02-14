#!/bin/bash
source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"


printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
                            type referred to by pkg_type"
    echo "  -r,  --release            Makes a release build"
    echo "  -h,  --help               Prints this help"
    echo
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo

    return 0
}


API_NAME=hipcc
PROJ_NAME=$API_NAME

TARGET="build"
MAKEOPTS="$DASH_JAY"
BUILD_TYPE="Debug"
SHARED_LIBS="ON"
BUILD_DIR=$(getBuildPath $API_NAME)
PACKAGE_DEB=$(getPackageRoot)/deb/$API_NAME
PACKAGE_RPM=$(getPackageRoot)/rpm/$API_NAME
PACKAGE_SRC="$(getSrcPath)"

VALID_STR=`getopt -o hcraswo:p: --long help,clean,release,address_sanitizer,static,outdir,wheel:,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-a  | --address_sanitizer)
            ack_and_ignore_asan ;;
        (-c  | --clean)
            TARGET="clean" ;;
        (-o  | --outdir)
            TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 1 ;;
        (-r  | --release)
            BUILD_TYPE="RelWithDebInfo" ;;
        (-s | --static)
            SHARED_LIBS="OFF" ;;
        (-h  | --help)
            printUsage ; exit 0 ;;
        --)     shift; break;;
        (*)
            echo "Invalid option [$1]" >&2; printUsage; exit 1 ;;
    esac
    shift 1
done

clean() {
    echo "Cleaning hipcc"
    rm -rf $BUILD_DIR

    echo "Cleaning up hipcc DEB and RPM packages"
    rm -rf $PACKAGE_DEB
    rm -rf $PACKAGE_RPM
}

copy_hipcc_sources() {
    echo "Clean up hipcc build folder"
    rm -rf "$PACKAGE_SRC/hipcc"
    echo "Copy hipcc sources"
    mkdir -p "$PACKAGE_SRC/hipcc"
    progressCopy "$HIPCC_ROOT" "$PACKAGE_SRC/hipcc"
}

build() {
    echo "Build hipcc binary"
    mkdir -p "$BUILD_DIR"

    pushd "$BUILD_DIR"
    if ! [ -e "$HIPCC_ROOT/CMakeLists.txt" ] ; then

        echo "No source for hipcc, exiting. this is not an error" >&2
        exit 0
    fi

    cmake \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        $(rocm_cmake_params) \
        $(rocm_common_cmake_params) \
        -DHIPCC_BACKWARD_COMPATIBILITY=OFF \
        -DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
        $HIPCC_ROOT
    popd

    cmake --build "$BUILD_DIR" -- $MAKEOPTS

    echo "Installing and Packaging hipcc"
    cmake --build "$BUILD_DIR" -- $MAKEOPTS install
    cmake --build "$BUILD_DIR" -- $MAKEOPTS package

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" $BUILD_DIR/hipcc*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" $BUILD_DIR/hipcc*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${PACKAGE_DEB};;
        ("rpm")
            echo ${PACKAGE_RPM};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1 ;;
    esac
    exit
}

case $TARGET in
    (clean) clean ;;
    (build) build ; copy_hipcc_sources ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
