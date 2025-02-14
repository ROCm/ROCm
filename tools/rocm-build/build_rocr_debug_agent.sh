#!/bin/bash
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
    echo "  -h,  --help             Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME=rocm-debug-agent
PROJ_NAME=$API_NAME
LIB_NAME=lib${API_NAME}.so
TARGET=build
MAKETARGET=deb
PACKAGE_ROOT=$(getPackageRoot)
PACKAGE_BIN="$(getBinPath)"
PACKAGE_LIB=$(getLibPath)
PACKAGE_INCLUDE=$(getIncludePath)
BUILD_DIR=$(getBuildPath $API_NAME)
PACKAGE_DEB=$(getPackageRoot)/deb/$API_NAME
PACKAGE_RPM=$(getPackageRoot)/rpm/$API_NAME
PACKAGE_PREFIX=$ROCM_INSTALL_PATH
BUILD_TYPE=Debug
MAKE_OPTS="$DASH_JAY -C"

TEST_PACKAGE_DIR="$(getBinPath)/rocm-debug-agent-test"
PACKAGE_UTILS=$(getUtilsPath)

BC_DIR="$PACKAGE_ROOT/lib/bitcode"
if [ -d "$BC_DIR" ] ; then
  export DEVICE_LIB_PATH=$BC_DIR
fi


SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
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
                BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                ack_and_skip_static ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)
                MAKETARGET="$2" ; shift 2;;
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

clean() {
    echo "Cleaning $PROJ_NAME"
    rm -rf $BUILD_DIR
    rm -rf $TEST_PACKAGE_DIR
    rm -rf $PACKAGE_DEB
    rm -rf $PACKAGE_RPM
    rm -rf $PACKAGE_ROOT/${PROJ_NAME}
    rm -rf $PACKAGE_LIB/${LIB_NAME}*
    rm -f $PACKAGE_UTILS/run_rocr_debug_agent_test.sh
}

build() {
    echo "Building $PROJ_NAME"

    PACKAGE_CMAKE="$(getCmakePath)"
    export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=$ROCM_PATH"
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        pushd "$BUILD_DIR"

        cmake $(rocm_cmake_params) \
            -DCMAKE_PREFIX_PATH="$PACKAGE_CMAKE/amd-dbgapi" \
            -DCMAKE_MODULE_PATH="$PACKAGE_CMAKE/hip" \
            $(rocm_common_cmake_params) \
            -DCMAKE_HIP_ARCHITECTURES=OFF \
            $ROCR_DEBUG_AGENT_ROOT

        popd
    fi
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS $BUILD_DIR
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS $BUILD_DIR install
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS $BUILD_DIR package

    mkdir -p $PACKAGE_LIB
    cp -R $BUILD_DIR/${LIB_NAME}* $PACKAGE_LIB

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "${PACKAGE_DEB}" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "${PACKAGE_RPM}" "$BUILD_DIR/${API_NAME}"*.rpm
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

verifyEnvSetup

case $TARGET in
    (clean) clean ;;
    (build) build ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $target" ;;
esac

echo "Operation complete"
