#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

PATH=${ROCM_PATH}/bin:$PATH
set_component_src rocFFT

build_rocfft() {
    echo "Start Build"

    cd $COMPONENT_SRC

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
    fi
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    fi

    CXX="${ROCM_PATH}/bin/hipcc" \
    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        -DUSE_HIP_CLANG=ON \
        -DHIP_COMPILER=clang  \
        -DBUILD_CLIENTS_SAMPLES=ON  \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_RIDER=ON  \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rocfft() {
    echo "Cleaning rocFFT build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocfft ;;
    outdir) print_output_directory ;;
    clean) clean_rocfft ;;
    *) die "Invalid target $TARGET" ;;
esac
