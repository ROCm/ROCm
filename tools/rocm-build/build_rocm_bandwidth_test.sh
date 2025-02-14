#!/bin/bash -x

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
        type referred to by pkg_type"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

TEST_BIN_DIR="$(getBinPath)"
TEST_NAME="rocm-bandwidth-test"
TEST_UTILS_DIR="$(getUtilsPath)"
TEST_SRC_DIR="rocm_bandwidth_test"
TEST_BLD_DIR="$(getBuildPath $TEST_SRC_DIR)"

ROCM_PKG_PREFIX="$ROCM_INSTALL_PATH"
TEST_PKG_ROOT="$(getPackageRoot)"
TEST_PKG_DEB="$(getPackageRoot)/deb/$TEST_SRC_DIR"
TEST_PKG_RPM="$(getPackageRoot)/rpm/$TEST_SRC_DIR"

ROCR_LIB_DIR="$(getPackageRoot)/lib"
ROCR_INC_DIR="$(getPackageRoot)/hsa/include"

RUN_SCRIPT=$(echo $(basename "${BASH_SOURCE}") | sed "s/build_/run_/")

TARGET="build"
MAKETARGET="all"
BUILD_TYPE="Debug"
MAKEARG="$DASH_JAY"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
PKGTYPE="deb"


VALID_STR=`getopt -o hcraso:p: --long help,clean,release,static,address_sanitizer,outdir:,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="Release" ; MAKEARG="$MAKEARG REL=1" ;  shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                ack_and_skip_static ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)
                MAKETARGET="$2" ; CPACKGEN="${2^^}" ; shift 2;;
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

clean_rocm_bandwidth_test() {
    echo "Cleaning $TEST_NAME"

    rm -rf $TEST_BLD_DIR
    rm -rf $TEST_PKG_DEB
    rm -rf $TEST_PKG_RPM
    rm -rf $TEST_BIN_DIR/$TEST_NAME
    rm -f  $TEST_UTILS_DIR/$RUN_SCRIPT
}

build_rocm_bandwidth_test() {

    echo "Building $TEST_NAME"

    if [ ! -d "$TEST_BLD_DIR" ]; then
        mkdir -p "$TEST_BLD_DIR"
        pushd "$TEST_BLD_DIR"

        cmake \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE"      \
            -DCMAKE_VERBOSE_MAKEFILE=1 \
            -DCMAKE_INSTALL_PREFIX="$TEST_PKG_ROOT" \
            -DCPACK_PACKAGING_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
            -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PATH" \
	    $(rocm_common_cmake_params) \
            -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
            -DROCM_PATCH_VERSION=$ROCM_LIBPATCH_VERSION \
            -DCMAKE_MODULE_PATH="$ROCM_BANDWIDTH_TEST_ROOT/cmake_modules" \
            -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
            "$ROCM_BANDWIDTH_TEST_ROOT"

        popd
    fi

    echo "Building $TEST_NAME"
    cmake --build "$TEST_BLD_DIR" -- $MAKEARG -C $TEST_BLD_DIR

    echo "Installing $TEST_NAME"
    cmake --build "$TEST_BLD_DIR" -- $MAKEARG -C $TEST_BLD_DIR install

    echo "Packaging $TEST_NAME"
    cmake --build "$TEST_BLD_DIR" -- $MAKEARG -C $TEST_BLD_DIR package

    mkdir -p "$TEST_BIN_DIR"
    echo "Copying $TEST_NAME to $TEST_BIN_DIR"
    progressCopy "$TEST_BLD_DIR/$TEST_NAME" "$TEST_BIN_DIR"

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$TEST_PKG_DEB" $TEST_BLD_DIR/*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$TEST_PKG_RPM" $TEST_BLD_DIR/*.rpm

}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${TEST_PKG_DEB};;
        ("rpm")
            echo ${TEST_PKG_RPM};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}
verifyEnvSetup

case $TARGET in
    (clean) clean_rocm_bandwidth_test ;;
    (build) build_rocm_bandwidth_test ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
