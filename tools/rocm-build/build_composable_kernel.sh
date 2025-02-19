#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src composable_kernel

GPU_ARCH_LIST="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"

build_miopen_ck() {
    echo "Start Building Composable Kernel"
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
       GPU_ARCH_LIST="gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+"
    else
       unset_asan_env_vars
       set_address_sanitizer_off
    fi

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        GPU_ARCH_LIST="gfx942"
        ack_and_skip_static
    fi

    PYTHON_VERSION_WORKAROUND=''
    echo "DISTRO_ID: ${DISTRO_ID}"
    if [ "$DISTRO_ID" = "rhel-8.8" ] || [ "$DISTRO_ID" = "sles-15.5" ] ; then
        EXTRA_PYTHON_PATH=/opt/Python-3.8.13
        PYTHON_VERSION_WORKAROUND="-DCK_USE_ALTERNATIVE_PYTHON=${EXTRA_PYTHON_PATH}/bin/python3.8"
        # For the python interpreter we need to export LD_LIBRARY_PATH.
        export LD_LIBRARY_PATH=${EXTRA_PYTHON_PATH}/lib:$LD_LIBRARY_PATH
    fi

    cd $COMPONENT_SRC
    mkdir "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    cmake \
        -DBUILD_DEV=OFF \
        "${rocm_math_common_cmake_params[@]}" \
        ${PYTHON_VERSION_WORKAROUND} \
        -DCPACK_GENERATOR="${PKGTYPE^^}" \
        -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
        -DCMAKE_C_COMPILER="${ROCM_PATH}/llvm/bin/clang" \
        ${LAUNCHER_FLAGS} \
        -DGPU_ARCHS="${GPU_ARCH_LIST}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS=" -O3 " \
        "$COMPONENT_SRC"

    cmake --build . -- -j${PROC} package
    cmake --build "$BUILD_DIR" -- install
    mkdir -p $PACKAGE_DIR && cp ./*.${PKGTYPE} $PACKAGE_DIR
}

unset_asan_env_vars() {
    ASAN_CMAKE_PARAMS="false"
    export ADDRESS_SANITIZER="OFF"
    export LD_LIBRARY_PATH=""
    export ASAN_OPTIONS=""
}

set_address_sanitizer_off() {
    export CFLAGS=""
    export CXXFLAGS=""
    export LDFLAGS=""
}

clean_miopen_ck() {
    echo "Cleaning MIOpen-CK build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_miopen_ck ;;
    outdir) print_output_directory ;;
    clean) clean_miopen_ck ;;
    *) die "Invalid target $TARGET" ;;
esac
