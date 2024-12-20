#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

build_release=true
set_component_src rocThrust

build_rocthrust() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    cd $COMPONENT_SRC

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
         ASAN_CMAKE_PARAMS="false"
    fi

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    init_rocm_common_cmake_params

    CXX=$(set_build_variables CXX)\
    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DCMAKE_MODULE_PATH="${ROCM_PATH}/lib/cmake/hip;${ROCM_PATH}/hip/cmake" \
        -DROCPRIM_ROOT="${ROCM_PATH}/rocprim" \
        -DBUILD_TEST=ON \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR


    show_build_cache_stats
}

clean_rocthrust() {
    echo "Cleaning rocThrust build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    (clean) clean_rocthrust ;;
    (build) build_rocthrust ;;
   (outdir) print_output_directory ;;
        (*) die "Invalid target $TARGET" ;;
esac
