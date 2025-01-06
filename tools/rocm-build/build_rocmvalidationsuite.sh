#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src ROCmValidationSuite

ROCM_RVS_LIB_RPATH="\$ORIGIN/.."

build_rocmvalidationsuite() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
    fi

    cd "${COMPONENT_SRC}"
    mkdir -p "$BUILD_DIR"
    init_rocm_common_cmake_params

    cmake \
        "${rocm_math_common_cmake_params[@]}" \
        -DFETCH_ROCMPATH_FROM_ROCMCORE=ON \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT="-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LIB_RPATH:$ROCM_RVS_LIB_RPATH" \
        -DRVS_BUILD_TESTS=FALSE \
        -B "$BUILD_DIR" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p "${PACKAGE_DIR}" && cp "${BUILD_DIR}"/*."${PKGTYPE}" "${PACKAGE_DIR}"

    show_build_cache_stats
}

clean_rocmvalidationsuite() {
    echo "Cleaning ROCmValidationSuite build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "${BUILD_DIR}" "${PACKAGE_DIR}"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocmvalidationsuite ;;
    outdir) print_output_directory ;;
    clean) clean_rocmvalidationsuite ;;
    *) die "Invalid target ${TARGET}" ;;
esac
