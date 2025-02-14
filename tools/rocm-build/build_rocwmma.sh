#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rocWMMA

build_rocwmma() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    if [ ! -e $COMPONENT_SRC/CMakeLists.txt ]; then
        echo "Skipping rocWMMA as source is not available"
        mkdir -p $COMPONENT_SRC
        exit 0
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
         ASAN_CMAKE_PARAMS="false"
    fi
    mkdir -p $BUILD_DIR && cd $BUILD_DIR

    init_rocm_common_cmake_params
    CXX=$(set_build_variables CXX)\
    cmake \
        "${rocm_math_common_cmake_params[@]}" \
        ${LAUNCHER_FLAGS} \
        -DROCWMMA_BUILD_VALIDATION_TESTS=ON \
        -DROCWMMA_VALIDATE_WITH_ROCBLAS=ON \
        -DROCWMMA_BUILD_BENCHMARK_TESTS=ON \
        -DROCWMMA_BENCHMARK_WITH_ROCBLAS=ON \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}

clean_rocwmma() {
    echo "Cleaning rocWMMA build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocwmma ;;
    outdir) print_output_directory ;;
    clean) clean_rocwmma ;;
    *) die "Invalid target $TARGET" ;;
esac
