#!/bin/bash
set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rocRAND

build_rocrand() {
    echo "Start build"

    SHARED_LIBS="ON"
    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        SHARED_LIBS="OFF"
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
    fi

    cd $COMPONENT_SRC && mkdir "$BUILD_DIR"

    remote_name=$(git remote show | head -n 1)
    [ "$remote_name" == "origin" ] || git remote rename "$remote_name" origin
    git submodule update --init --force

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101"
    fi

    init_rocm_common_cmake_params

    CXX=$(set_build_variables CXX)\
    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        -DBUILD_TEST=ON \
        -DBUILD_BENCHMARK=ON \
        -DBUILD_CRUSH_TEST=ON \
        -DDEPENDENCIES_FORCE_DOWNLOAD=OFF \
        -DHIP_COMPILER=clang \
        -DCMAKE_MODULE_PATH="${ROCM_PATH}/lib/cmake/hip" \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        -B "${BUILD_DIR}" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/  && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rocrand() {
    echo "Cleaning rocRAND build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocrand ;;
    outdir) print_output_directory ;;
    clean) clean_rocrand ;;
    *) die "Invalid target $TARGET" ;;
esac
