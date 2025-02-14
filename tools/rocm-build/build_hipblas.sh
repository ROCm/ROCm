#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src hipBLAS

build_hipblas() {
    echo "Start build"

    CXX="g++"
    CXX_FLAG=
    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        CXX="amdclang++"
        CXX_FLAG="-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++"
    fi

    CLIENTS_SAMPLES="ON"
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
       CLIENTS_SAMPLES="OFF"
    fi

    SHARED_LIBS="ON"

    echo "C compiler: $CC"
    echo "CXX compiler: $CXX"
    echo "FC compiler: $FC"

    cd $COMPONENT_SRC
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       rebuild_lapack
    fi

    init_rocm_common_cmake_params
    cmake \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DUSE_CUDA=OFF \
        -DBUILD_SHARED_LIBS=$SHARED_LIBS \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DBUILD_CLIENTS_SAMPLES="${CLIENTS_SAMPLES}" \
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

clean_hipblas() {
    echo "Cleaning hipBLAS build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_hipblas ;;
    outdir) print_output_directory ;;
    clean) clean_hipblas ;;
    *) die "Invalid target $TARGET" ;;
esac
