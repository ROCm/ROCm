#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src hipfort

build_hipfort() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake \
        -DCPACK_PACKAGING_INSTALL_PREFIX=${ROCM_PATH}\
        -DHIPFORT_INSTALL_DIR="${ROCM_PATH}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}/llvm;${ROCM_PATH}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCPACK_SET_DESTDIR="OFF" \
        -DCPACK_RPM_PACKAGE_RELOCATABLE="ON" \
        -DHIPFORT_COMPILER="${ROCM_PATH}/${ROCM_LLVMDIR}/bin/flang" \
        -DCMAKE_Fortran_FLAGS="-Mfree" \
        -DHIPFORT_COMPILER_FLAGS="-cpp" \
        -DCMAKE_Fortran_FLAGS_DEBUG="" \
        ${LAUNCHER_FLAGS} \
        -DROCM_SYMLINK_LIBS=OFF \
	-DCMAKE_INSTALL_PREFIX=${ROCM_PATH} \
        -DHIPFORT_AR="${ROCM_PATH}/${ROCM_LLVMDIR}/bin/llvm-ar" \
        -DHIPFORT_RANLIB="${ROCM_PATH}/${ROCM_LLVMDIR}/bin/llvm-ranlib" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_hipfort() {
    echo "Cleaning hipFORT build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_hipfort ;;
    outdir) print_output_directory ;;
    clean) clean_hipfort ;;
    *) die "Invalid target $TARGET" ;;
esac
