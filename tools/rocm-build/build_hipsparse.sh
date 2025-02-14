#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

PATH=${ROCM_PATH}/bin:$PATH
set_component_src hipSPARSE

build_hipsparse() {
    echo "Start build"

    CXX="g++"
    CXX_FLAG=

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        CXX="${ROCM_PATH}/llvm/bin/clang++"
        CXX_FLAG="-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++"
    fi

    cd $COMPONENT_SRC

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
        set_asan_env_vars
        set_address_sanitizer_on
    fi

    SHARED_LIBS="ON"
    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        SHARED_LIBS="OFF"
    fi

    echo "C compiler: $CC"
    echo "CXX compiler: $CXX"

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        -DUSE_CUDA=OFF  \
        -DBUILD_CLIENTS_SAMPLES=ON \
        -DBUILD_CLIENTS_TESTS=ON \
        -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} \
        -DCMAKE_MODULE_PATH="${ROCM_PATH}/lib/cmake/hip;${ROCM_PATH}/hip/cmake"  \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        ${CXX_FLAG} \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    show_build_cache_stats
}

clean_hipsparse() {
    echo "Cleaning hipSPARSE build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_hipsparse ;;
    outdir) print_output_directory ;;
    clean) clean_hipsparse ;;
    *) die "Invalid target $TARGET" ;;
esac
