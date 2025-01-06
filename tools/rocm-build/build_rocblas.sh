#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rocBLAS
DEPS_DIR=${HOME}/rocblas

stage2_command_args "$@"

build_rocblas() {
    echo "Start build"

    SHARED_LIBS="ON"
    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        SHARED_LIBS="OFF"
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
       export ASAN_OPTIONS="detect_leaks=0:verify_asan_link_order=0"
    fi

    cd $COMPONENT_SRC

    mkdir -p $DEPS_DIR && cp -r /usr/blis $DEPS_DIR
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    fi
    init_rocm_common_cmake_params

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=toolchain-linux.cmake \
        -DBUILD_DIR="${BUILD_DIR}" \
	"${rocm_math_common_cmake_params[@]}" \
        -DROCM_DIR="${ROCM_PATH}" \
        ${LAUNCHER_FLAGS} \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        -DCMAKE_PREFIX_PATH="${DEPS_DIR};${ROCM_PATH}" \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DBUILD_CLIENTS_SAMPLES=ON \
        -DLINK_BLIS=ON \
        -DAMDGPU_TARGETS="${GPU_TARGETS}" \
        -DTensile_CODE_OBJECT_VERSION=default \
        -DTensile_LOGIC=asm_full \
        -DTensile_SEPARATE_ARCHITECTURES=ON \
        -DTensile_LAZY_LIBRARY_LOADING=ON \
        -DTensile_LIBRARY_FORMAT=msgpack \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        -DTENSILE_VENV_UPGRADE_PIP=ON \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && rm -rf ./library/src/build_tmp && find -name '*.o' -delete

    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rocblas() {
    echo "Cleaning rocBLAS build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

case $TARGET in
    build) build_rocblas ;;
    outdir) print_output_directory ;;
    clean) clean_rocblas ;;
    *) die "Invalid target $TARGET" ;;
esac
