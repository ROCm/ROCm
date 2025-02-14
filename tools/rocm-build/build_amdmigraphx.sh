#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src AMDMIGraphX

build_amdmigraphx() {
    echo "Start build"

    cd $COMPONENT_SRC

    if ! command -v rbuild &> /dev/null; then
        pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
    fi

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        GPU_TARGETS="gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1102;gfx942;gfx1200;gfx1201"
    fi
    init_rocm_common_cmake_params

    mkdir -p ${BUILD_DIR} && rm -rf ${BUILD_DIR}/* && mkdir -p ${HOME}/amdmigraphx && rm -rf ${HOME}/amdmigraphx/*
    rbuild package -d "${HOME}/amdmigraphx" -B "${BUILD_DIR}" \
        --cxx="${ROCM_PATH}/llvm/bin/clang++" \
        --cc="${ROCM_PATH}/llvm/bin/clang" \
        "${rocm_math_common_cmake_params[@]}" \
        -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LIB_RPATH" \
	    -DGPU_TARGETS="${GPU_TARGETS}" \
        -DCMAKE_INSTALL_RPATH=""

    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    cd $BUILD_DIR && cmake --build . -- install -j${PROC}

    show_build_cache_stats
}

clean_amdmigraphx() {
    echo "Cleaning AMDMIGraphX build directory: ${BUILD_DIR} ${DEPS_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$DEPS_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_amdmigraphx ;;
    outdir) print_output_directory ;;
    clean) clean_amdmigraphx ;;
    *) die "Invalid target $TARGET" ;;
esac
