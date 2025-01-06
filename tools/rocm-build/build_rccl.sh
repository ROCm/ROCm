#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rccl

ENABLE_ADDRESS_SANITIZER=false

build_rccl() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    mkdir -p $ROCM_PATH/.info/
    echo $ROCM_VERSION | tee $ROCM_PATH/.info/version

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
    fi

    mkdir -p $BUILD_DIR && cd $BUILD_DIR

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx940;gfx941;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    fi

    init_rocm_common_cmake_params

    CC=${ROCM_PATH}/bin/amdclang \
    CXX=$(set_build_variables CXX) \
    cmake \
        "${rocm_math_common_cmake_params[@]}" \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        -DHIP_COMPILER=clang \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH};${ROCM_PATH}/share/rocm/cmake/" \
        ${LAUNCHER_FLAGS} \
        -DCPACK_GENERATOR="${PKGTYPE^^}" \
        -DROCM_PATCH_VERSION=$ROCM_LIBPATCH_VERSION \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        -DBUILD_TESTS=ON \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_rccl() {
    echo "Cleaning rccl build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rccl ;;
    outdir) print_output_directory ;;
    clean) clean_rccl ;;
    *) die "Invalid target $TARGET" ;;
esac
