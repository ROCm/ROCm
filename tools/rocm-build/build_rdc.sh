#!/bin/bash

source "$(dirname "$BASH_SOURCE")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename $0) [-c|-r|-h] [makeopts]"
    echo
    echo "Options:"
    echo "  -c,  --clean            Removes all rdc build artifacts, except grpc"
    echo "  -g,  --clean_grpc       Removes the grpc files and artifacts"
    echo "  -r,  --release          Build release version of RDC (default is debug)"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
        type referred to by pkg_type"
    echo "  -s,  --static           Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo "  -h,  --help             Prints this help"
    echo
    return 0
}

PACKAGE_ROOT="$(getPackageRoot)"
RDC_BUILD_DIR=$(getBuildPath rdc)
GRPC_BUILD_DIR=$(getBuildPath grpc)
TARGET="build"
PACKAGE_LIB=$(getLibPath)
PACKAGE_INCLUDE="$(getIncludePath)"
PACKAGE_BIN="$(getBinPath)"
RDC_PACKAGE_DEB_DIR="$PACKAGE_ROOT/deb/rdc"
RDC_PACKAGE_RPM_DIR="$PACKAGE_ROOT/rpm/rdc"
BUILD_TYPE="Debug"
MAKETARGET="deb"
MAKEARG="$DASH_JAY O=$RDC_BUILD_DIR"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
CLEAN_GRPC="no"
PKGTYPE="deb"
RDC_MAKE_OPTS="$DASH_JAY O=$RDC_BUILD_DIR -C $RDC_BUILD_DIR"
BUILD_DOCS="no"
RDC_PKG_NAME_ROOT="rdc"
RDC_PKG_NAME="${RDC_PKG_NAME_ROOT}"
GRPC_PROTOC_ROOT="${RDC_BUILD_DIR}/grpc"
GRPC_SEARCH_ROOT="/usr/grpc"
GRPC_DESIRED_VERSION="1.61.0"

RDC_LIB_RPATH='$ORIGIN'
RDC_LIB_RPATH=$RDC_LIB_RPATH:'$ORIGIN/..'
RDC_LIB_RPATH=$RDC_LIB_RPATH:'$ORIGIN/rdc/grpc/lib'
RDC_LIB_RPATH=$RDC_LIB_RPATH:'$ORIGIN/grpc/lib'
RDC_EXE_RPATH='$ORIGIN/../lib'
RDC_EXE_RPATH=$RDC_EXE_RPATH:'$ORIGIN/../lib/rdc/grpc/lib'

VALID_STR=`getopt -o hcgradso:p: --long help,clean,clean_grpc,release,documentation,static,address_sanitizer,outdir:,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-g | --clean_grpc)
                TARGET="clean_grpc" ; shift ;;
        (-r | --release)
                BUILD_TYPE="Release" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-d | --documentation )
                BUILD_DOCS="yes" ;;
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


clean_rdc() {
    rm -rf "$RDC_BUILD_DIR"
    rm -rf "$RDC_PACKAGE_DEB_DIR"
    rm -rf "$RDC_PACKAGE_RPM_DIR"
    rm -rf "$RDC_BUILD_DIR/rdc"

    rm -rf "$PACKAGE_INCLUDE/rdc"
    rm -f $PACKAGE_LIB/librdc*
    rm -f $PACKAGE_BIN/rdci
    return 0
}

clean_grpc() {
    rm -rf "$GRPC_BUILD_DIR"
}

find_grpc() {
    grep -s -F "$GRPC_DESIRED_VERSION" ${GRPC_SEARCH_ROOT}/*/cmake/grpc/gRPCConfigVersion.cmake &&
        GRPC_PROTOC_ROOT=$GRPC_SEARCH_ROOT
}

rdc_backwards_compat_cmake_params() {
    grep -q "RDC_CLIENT_INSTALL_PREFIX" "$RDC_ROOT/CMakeLists.txt" &&
        echo "-DRDC_CLIENT_INSTALL_PREFIX=$PACKAGE_ROOT"
}

build_rdc() {
    if ! find_grpc; then
        echo "ERROR: GRPC SEARCH FAILED!"
        echo "You are expected to have gRPC [${GRPC_DESIRED_VERSION}] in [${GRPC_SEARCH_ROOT}]"
        # Compiling gRPC as part of the RDC build takes too long and times out the build job
        return 1
    fi
    echo "gRPC [${GRPC_DESIRED_VERSION}] found!"

    echo "Building RDC"
    echo "RDC_BUILD_DIR: ${RDC_BUILD_DIR}"
    echo "GRPC_PROTOC_ROOT: ${GRPC_PROTOC_ROOT}"

    export LD_PRELOAD="$ASAN_LIB_PATH"

    if [ ! -d "$RDC_BUILD_DIR/rdc_libs" ]; then
        mkdir -p $RDC_BUILD_DIR
        pushd $RDC_BUILD_DIR
        cmake \
            -DGRPC_ROOT="$GRPC_PROTOC_ROOT" \
            -DGRPC_DESIRED_VERSION="$GRPC_DESIRED_VERSION" \
            -DCMAKE_MODULE_PATH="$RDC_ROOT/cmake_modules" \
            $(rocm_cmake_params) \
            $(rdc_backwards_compat_cmake_params) \
            $(rocm_common_cmake_params) \
            -DROCM_DIR=$ROCM_INSTALL_PATH \
            -DRDC_PACKAGE="${RDC_PKG_NAME}" \
            -DCPACK_PACKAGE_VERSION_MAJOR="1" \
            -DCPACK_PACKAGE_VERSION_MINOR="$ROCM_LIBPATCH_VERSION" \
            -DCPACK_PACKAGE_VERSION_PATCH="0" \
            -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
            -DBUILD_TESTS=ON \
            -DCMAKE_SKIP_BUILD_RPATH=TRUE \
            -DCMAKE_EXE_LINKER_FLAGS_INIT="-Wl,--no-as-needed,-z,origin,--enable-new-dtags,--build-id=sha1,--rpath,$RDC_EXE_RPATH" \
            -DCMAKE_SHARED_LINKER_FLAGS_INIT="-Wl,--no-as-needed,-z,origin,--enable-new-dtags,--build-id=sha1,--rpath,$RDC_LIB_RPATH" \
            "$RDC_ROOT"
        popd
    fi
    echo "Making rdc package:"
    cmake --build "$RDC_BUILD_DIR" -- $RDC_MAKE_OPTS
    cmake --build "$RDC_BUILD_DIR" -- $RDC_MAKE_OPTS install

    unset LD_PRELOAD
    cmake --build "$RDC_BUILD_DIR" -- $RDC_MAKE_OPTS package

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$RDC_PACKAGE_DEB_DIR" "$RDC_BUILD_DIR/$RDC_PKG_NAME"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$RDC_PACKAGE_RPM_DIR" "$RDC_BUILD_DIR/$RDC_PKG_NAME"*.rpm

    if [ ! -e $ROCM_INSTALL_PATH/include/rdc/rdc.h ]; then
      cp -r "$ROCM_INSTALL_PATH/rdc/lib/." "$PACKAGE_LIB"
      cp -r "$ROCM_INSTALL_PATH/rdc/bin/." "$PACKAGE_BIN"
      cp -r "$ROCM_INSTALL_PATH/rdc/include/." "$PACKAGE_INCLUDE"
    fi

    if [ "$BUILD_DOCS" = "yes" ]; then
      echo "Building Docs"
      cmake --build "$RDC_BUILD_DIR" -- $RDC_MAKE_OPTS doc
      pushd $RDC_BUILD_DIR/latex
      cmake --build . --
      mv refman.pdf "$ROCM_INSTALL_PATH/rdc/RDC_Manual.pdf"
      popd
    fi
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${RDC_PACKAGE_DEB_DIR};;
        ("rpm")
            echo ${RDC_PACKAGE_RPM_DIR};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

verifyEnvSetup

case $TARGET in
    (clean) clean_rdc ;;
    (clean_grpc) clean_grpc ;;
    (build) build_rdc ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
