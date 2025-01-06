#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [-c|-r|-32|-h] [makeopts]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Removes all rocm_smi build artifacts"
    echo "  -r,  --release            Build non-debug version rocm_smi (default is debug)"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of type referred to by pkg_type"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -32,                      Build 32b version (default is 64b)"
    echo "  -h,  --help             Prints this help"
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

PROJ_NAME="rsmi"
PACKAGE_ROOT="$(getPackageRoot)"
TARGET="build"

PACKAGE_LIB=$(getLibPath)
PACKAGE_INCLUDE="$(getIncludePath)"

RSMI_BUILD_DIR=$(getBuildPath rsmi)
RSMI_PACKAGE_DEB_DIR="$(getPackageRoot)/deb/$PROJ_NAME"
RSMI_PACKAGE_RPM_DIR="$(getPackageRoot)/rpm/$PROJ_NAME"
RSMI_BUILD_TYPE="debug"
BUILD_TYPE="Debug"

MAKETARGET="deb"
MAKEARG="$DASH_JAY O=$RSMI_BUILD_DIR"
RSMI_MAKE_OPTS="$DASH_JAY O=$RSMI_BUILD_DIR -C $RSMI_BUILD_DIR"
ROCM_SMI_BLD_BITS=64
RSMI_PKG_NAME_ROOT="rocm-smi-lib"
RSMI_PKG_NAME="${RSMI_PKG_NAME_ROOT}${ROCM_SMI_BLD_BITS}"
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
                BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on
                # TODO - support standard option of passing cmake environment vars - CFLAGS,CXXFLAGS etc., to enable address sanitizer
                ADDRESS_SANITIZER=true ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)
                MAKETARGET="$2" ; shift 2;;
        (-32)
                ROCM_SMI_BLD_BITS="32"; shift ;;
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

clean_rsmi() {
    rm -rf "$RSMI_BUILD_DIR"
    rm -rf "$RSMI_PACKAGE_DEB_DIR"
    rm -rf "$RSMI_PACKAGE_RPM_DIR"
    rm -rf "$PACKAGE_ROOT/rocm_smi"
    rm -rf "$PACKAGE_INCLUDE/rocm_smi"
    rm -f $PACKAGE_LIB/librocm_smi${ROCM_SMI_BLD_BITS}.*
    rm -f $PACKAGE_LIB/librocm_smi${ROCM_SMI_BLD_BITS}.*
    return 0
}

build_rsmi() {
    echo "Building RSMI"
    echo "RSMI_BUILD_DIR: ${RSMI_BUILD_DIR}"
    if [ ! -d "$RSMI_BUILD_DIR" ]; then
        mkdir -p $RSMI_BUILD_DIR
        pushd $RSMI_BUILD_DIR
        print_lib_type $SHARED_LIBS

        cmake \
            -DCMAKE_MODULE_PATH="$ROCM_SMI_LIB_ROOT/cmake_modules" \
            -DBUILD_SHARED_LIBS=$SHARED_LIBS \
	    $(rocm_common_cmake_params) \
            $(rocm_cmake_params) \
            -DENABLE_LDCONFIG=OFF \
            -DROCM_SMI_PACKAGE="${RSMI_PKG_NAME}" \
            -DCPACK_PACKAGE_VERSION_MAJOR="1" \
            -DCPACK_PACKAGE_VERSION_MINOR="$ROCM_LIBPATCH_VERSION" \
            -DCPACK_PACKAGE_VERSION_PATCH="0" \
            -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
            -DBUILD_TESTS=ON \
            "$ROCM_SMI_LIB_ROOT"
        popd
    fi

    echo "Making rocm_smi package:"
    cmake --build "$RSMI_BUILD_DIR" -- $RSMI_MAKE_OPTS
    cmake --build "$RSMI_BUILD_DIR" -- $RSMI_MAKE_OPTS install
    cmake --build "$RSMI_BUILD_DIR" -- $RSMI_MAKE_OPTS package

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$RSMI_PACKAGE_DEB_DIR" $RSMI_BUILD_DIR/*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$RSMI_PACKAGE_RPM_DIR" $RSMI_BUILD_DIR/*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${RSMI_PACKAGE_DEB_DIR};;
        ("rpm")
            echo ${RSMI_PACKAGE_RPM_DIR};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

verifyEnvSetup

case $TARGET in
    (clean) clean_rsmi ;;
    (build) build_rsmi ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
exit 0
