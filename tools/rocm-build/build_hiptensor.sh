#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src hipTensor

build_hiptensor() {
    echo "Start build hipTensor"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
        set_asan_env_vars
        set_address_sanitizer_on
    fi

    cd "$COMPONENT_SRC"
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942"
    fi

    cmake \
        -B "${BUILD_DIR}" \
        "${rocm_math_common_cmake_params[@]}" \
        $(set_build_variables CMAKE_C_CXX) \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        ${LAUNCHER_FLAGS} \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p "$PACKAGE_DIR" && cp ${BUILD_DIR}/*.${PKGTYPE} "$PACKAGE_DIR"

    show_build_cache_stats
}

clean_hiptensor() {
    echo "Cleaning hipTensor build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_hiptensor ;;
    outdir) print_output_directory ;;
    clean) clean_hiptensor ;;
    *) die "Invalid target $TARGET" ;;
esac
