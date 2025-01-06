#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rpp
DEPS_DIR="$RPP_DEPS_LOCATION"

LLVM_LIBDIR="${ROCM_PATH}/llvm/lib"
ROCM_LLVM_LIB_RPATH="\$ORIGIN/llvm/lib"

rpp_specific_cmake_params() {
    local rpp_cmake_params
    if [ "${ASAN_CMAKE_PARAMS}" == "true" ] ; then
        rpp_cmake_params="-DCMAKE_EXE_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_ASAN_EXE_RPATH:$LLVM_LIBDIR"
    else
        rpp_cmake_params=""
    fi
    printf '%s ' "${rpp_cmake_params}"
}

build_rpp() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    if [ ! -e $COMPONENT_SRC/CMakeLists.txt ]; then
        echo "Skipping RPP build as source is not available"
        mkdir -p $COMPONENT_SRC
        exit 0
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
        set_asan_env_vars
        set_address_sanitizer_on
    fi

    mkdir -p $BUILD_DIR && cd $BUILD_DIR

    init_rocm_common_cmake_params

    cmake \
        "${rocm_math_common_cmake_params[@]}" \
        ${LAUNCHER_FLAGS} \
        -DBACKEND=HIP \
        -DCMAKE_INSTALL_LIBDIR=$(getInstallLibDir) \
        $(rpp_specific_cmake_params) \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fno-openmp-implicit-rpath -Wl,--enable-new-dtags,--build-id=sha1,--rpath,${ROCM_LIB_RPATH}:${DEPS_DIR}/lib:${ROCM_LLVM_LIB_RPATH}" \
        -DCMAKE_PREFIX_PATH="${DEPS_DIR};${ROCM_PATH}" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cpack -G ${PKGTYPE^^}

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR
    cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}

clean_rpp() {
    echo "Cleaning rpp build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rpp ;;
    outdir) print_output_directory ;;
    clean) clean_rpp ;;
    *) die "Invalid target $TARGET" ;;
esac
