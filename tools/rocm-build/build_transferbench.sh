#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src TransferBench

build_transferbench() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    sed -i 's/^\(\s*set\s*(CMAKE_RUNTIME_OUTPUT_DIRECTORY.*\)$/#\1/'  "${COMPONENT_SRC}/CMakeLists.txt"

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    CXX="$ROCM_PATH"/bin/hipcc \
    cmake "${rocm_math_common_cmake_params[@]}" "$COMPONENT_SRC"
    make package

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR
    cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}

clean_transferbench() {
    echo "Cleaning TransferBench build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_transferbench ;;
    outdir) print_output_directory ;;
    clean) clean_transferbench ;;
    *) die "Invalid target $TARGET" ;;
esac
