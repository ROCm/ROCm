#!/bin/bash
source "$(dirname $BASH_SOURCE)/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: ${BASH_SOURCE##*/} [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
    type referred to by pkg_type"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo "  -h,  --help               Prints this help"
    echo

    return 0
}

API_NAME="roctracer"
PROJ_NAME="$API_NAME"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_INCLUDE="$(getIncludePath)"
BUILD_DIR="$(getBuildPath $API_NAME)"
PACKAGE_DEB="$(getPackageRoot)/deb/$API_NAME"
PACKAGE_RPM="$(getPackageRoot)/rpm/$API_NAME"
PACKAGE_PREFIX="$ROCM_INSTALL_PATH"

export HIP_PATH="${ROCM_INSTALL_PATH}"
export HCC_HOME="${PACKAGE_ROOT}/hcc"

TARGET="build"
BUILD_TYPE="Debug"
MAKE_OPTS="$DASH_JAY -C $BUILD_DIR"
HIP_VDI=1
SHARED_LIBS="ON"
CLEAN_OR_OUT=0
MAKETARGET="deb"
PKGTYPE="deb"

GPU_LIST="gfx900;gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1031;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"

VALID_STR=$(getopt -o hcraso: --long help,clean,release,static,address_sanitizer,outdir: -- "$@")
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
        set_asan_env_vars
        set_address_sanitizer_on
        shift
        ;;
    -s | --static)
        ack_and_skip_static
        shift
        ;;
    -o | --outdir)
        TARGET="outdir"
        PKGTYPE=$2
        OUT_DIR_SPECIFIED=1
        ((CLEAN_OR_OUT |= 2))
        shift 2
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
    rm -rf "$PACKAGE_INCLUDE/${PROJ_NAME}"
    rm -rf "$PACKAGE_ROOT/${PROJ_NAME}"
}

copy_libs_to_workspace() {
    if [ "$ASAN_CMAKE_PARAMS" != "true" ]; then
        cp -R "${ROCM_INSTALL_PATH}/lib/roctracer/" "${OUT_DIR}/lib/roctracer/"
    fi
}

build() {
    echo "Building $PROJ_NAME"
    PACKAGE_CMAKE="$(getCmakePath)"
    export ROCM_PATH="${ROCM_INSTALL_PATH}"
    export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=$ROCM_PATH"

    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        pushd "$BUILD_DIR"
        print_lib_type $SHARED_LIBS

        export HIPCC_COMPILE_FLAGS_APPEND="--rocm-path=$ROCM_PATH --offload-arch=gfx900 --offload-arch=gfx906  --offload-arch=gfx908 \
                                                            --offload-arch=gfx90a --offload-arch=gfx940 --offload-arch=gfx941 \
                                                            --offload-arch=gfx942 --offload-arch=gfx1030 --offload-arch=gfx1031 \
                                                            --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 \
                                                            --offload-arch=gfx1200 --offload-arch=gfx1201"
        cmake $(rocm_cmake_params) \
            -DCPACK_PACKAGING_INSTALL_PREFIX="$PACKAGE_PREFIX/$PROJ_NAME" \
            -DCMAKE_MODULE_PATH="$PACKAGE_CMAKE/hip" \
            -DCMAKE_HIP_ARCHITECTURES=OFF \
            -DBUILD_SHARED_LIBS=$SHARED_LIBS \
            $(rocm_common_cmake_params) \
            -DCMAKE_HIP_ARCHITECTURES=OFF \
            -DENABLE_LDCONFIG=OFF \
            -DROCM_ROOT_DIR="$ROCM_INSTALL_PATH" \
            -DHIP_VDI="$HIP_VDI" \
            -DROCM_RPATH="$ROCM_RPATH" \
            -DGPU_TARGETS="$GPU_LIST" \
            -DCPACK_OBJCOPY_EXECUTABLE="${ROCM_INSTALL_PATH}/llvm/bin/llvm-objcopy" \
            -DCPACK_READELF_EXECUTABLE="${ROCM_INSTALL_PATH}/llvm/bin/llvm-readelf" \
            -DCPACK_STRIP_EXECUTABLE="${ROCM_INSTALL_PATH}/llvm/bin/llvm-strip" \
            -DCPACK_OBJDUMP_EXECUTABLE="${ROCM_INSTALL_PATH}/llvm/bin/llvm-objdump" \
            "$ROCTRACER_ROOT"

        popd
    fi
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS mytest
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS doc
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS install
    cmake --build "$BUILD_DIR" -- $MAKE_OPTS package

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" "$BUILD_DIR/${API_NAME}"*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
    "deb")
        echo ${PACKAGE_DEB}
        ;;
    "rpm")
        echo ${PACKAGE_RPM}
        ;;
    *)
        echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2
        exit 1
        ;;
    esac
    exit
}
verifyEnvSetup

case $TARGET in
clean) clean ;;
build) build && copy_libs_to_workspace ;;
outdir) print_output_directory ;;
*) die "$BASH_SOURCE Invalid target $TARGET - exiting" ;;
esac

echo "Operation complete"
