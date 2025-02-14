#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src MIVisionX
BUILD_DEV=ON

build_mivisionx() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
       BUILD_DEV=OFF
    fi

    init_rocm_common_cmake_params

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    fi

    cmake \
        "${rocm_math_common_cmake_params[@]}" \
        -DROCM_PATH="$ROCM_PATH" \
        -DBUILD_DEV=$BUILD_DEV \
        -DCMAKE_INSTALL_LIBDIR=$(getInstallLibDir) \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        -DROCM_DEP_ROCMCORE=ON \
        -DROCAL_PYTHON=OFF \
        ${LAUNCHER_FLAGS} \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cpack -G ${PKGTYPE^^}

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_mivisionx() {
    echo "Cleaning MIVisionX build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_mivisionx ;;
    outdir) print_output_directory ;;
    clean) clean_mivisionx ;;
    *) die "Invalid target $TARGET" ;;
esac
