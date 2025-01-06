#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rocPRIM

build_rocprim() {
    echo "Start build"

    cd $COMPONENT_SRC
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
       ASAN_CMAKE_PARAMS="false"
    fi

    SHARED_LIBS="ON"
    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        SHARED_LIBS="OFF"
    fi

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    fi

    init_rocm_common_cmake_params
    CXX="${ROCM_PATH}/bin/hipcc" \
    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_TEST=ON \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        -DCMAKE_MODULE_PATH="${ROCM_PATH}/lib/cmake/hip;${ROCM_PATH}/hip/cmake" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rocprim() {
    echo "Cleaning rocPRIM build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocprim ;;
    outdir) print_output_directory ;;
    clean) clean_rocprim ;;
    *) die "Invalid target $TARGET" ;;
esac
