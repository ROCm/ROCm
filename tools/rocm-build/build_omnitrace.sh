#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: ${BASH_SOURCE##*/} [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
                                      type referred to by pkg_type"
    echo "  -w,  --wheel              Creates python wheel package of omnitrace.
                                      It needs to be used along with -r option"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME="omnitrace"
PROJ_NAME="$API_NAME"
LIB_NAME="lib${API_NAME}"
TARGET="build"
MAKETARGET="deb"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_LIB="$(getLibPath)"
BUILD_DIR="$(getBuildPath $API_NAME)"
PACKAGE_DEB="$(getPackageRoot)/deb/$API_NAME"
PACKAGE_RPM="$(getPackageRoot)/rpm/$API_NAME"
BUILD_TYPE="Debug"
MAKE_OPTS="-j 8"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0
MAKETARGET="deb"
PKGTYPE="deb"
ASAN=0

#parse the arguments
VALID_STR=$(getopt -o hcraso:p:w --long help,clean,release,address_sanitizer,static,outdir:,package:,wheel -- "$@")
eval set -- "$VALID_STR"

while true; do
    case "$1" in
    -h | --help)
        printUsage
        exit 0
        ;;
    -c | --clean)
        TARGET="clean"
        ((CLEAN_OR_OUT |= 1))
        shift
        ;;
    -r | --release)
        BUILD_TYPE="RelWithDebInfo"
        shift
        ;;
    -a | --address_sanitizer)
        ack_and_ignore_asan

        ASAN=1
        shift
        ;;
    -s | --static)
        SHARED_LIBS="OFF"
        shift
        ;;
    -o | --outdir)
        TARGET="outdir"
        PKGTYPE=$2
        ((CLEAN_OR_OUT |= 2))
        shift 2
        ;;
    -p | --package)
        MAKETARGET="$2"
        shift 2
        ;;
    -w | --wheel)
	   echo "omnitrace: wheel build option accepted and ignored"
       shift
       ;;
    --)
        shift
        break
        ;;
    *)
        echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] " >&2
        exit 20
        ;;
    esac

done

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
    print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
    exit $RET_CONFLICT
fi

clean() {
    echo "Cleaning $PROJ_NAME"
    rm -rf "$BUILD_DIR"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$PACKAGE_ROOT/${PROJ_NAME:?}"
    rm -rf "$PACKAGE_LIB/${LIB_NAME:?}"*
}

build_omnitrace() {
    echo "Building $PROJ_NAME"
    if [ "$DISTRO_ID" = "mariner-2.0" ] || [ "$DISTRO_ID" = "ubuntu-24.04" ] || [ "$DISTRO_ID" = "azurelinux-3.0" ]; then
        echo "Skip make and uploading packages for Omnitrace on \"${DISTRO_ID}\" distro"
        exit 0
    fi

    if [ $ASAN == 1 ]; then
        echo "Skip make and uploading packages for Omnitrace on ASAN build"
        exit 0
    fi
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        echo "Created build directory: $BUILD_DIR"
    fi

    echo "Build directory: $BUILD_DIR"
    pushd "$BUILD_DIR" || exit
    print_lib_type $SHARED_LIBS

    echo "ROCm CMake Params: $(rocm_cmake_params)"
    echo "ROCm Common CMake Params: $(rocm_common_cmake_params)"


    if [ $ASAN == 1 ]; then
        echo "Address Sanitizer path"

    else
        cmake \
            $(rocm_cmake_params) \
            $(rocm_common_cmake_params) \
            -DOMNITRACE_BUILD_{LIBUNWIND,DYNINST}=ON \
            -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON \
            "$OMNITRACE_ROOT"
    fi


    popd || exit

    echo "Make Options: $MAKE_OPTS"
    cmake --build "$BUILD_DIR" --target all -- $MAKE_OPTS
    cmake --build "$BUILD_DIR" --target install -- $MAKE_OPTS
    cmake --build "$BUILD_DIR" --target package -- $MAKE_OPTS

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" "$BUILD_DIR/${API_NAME}"*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
    "deb")
        echo "${PACKAGE_DEB}"
        ;;
    "rpm")
        echo "${PACKAGE_RPM}"
        ;;
    *)
        echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2
        exit 1
        ;;
    esac
    exit
}

verifyEnvSetup

case "$TARGET" in
clean) clean ;;
build) build_omnitrace ;;
outdir) print_output_directory ;;
*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
