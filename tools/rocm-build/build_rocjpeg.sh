#!/bin/bash
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"
set_component_src rocJPEG
BUILD_DEV=ON
build_rocjpeg() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    # python3 ../rocJPEG-setup.py

    cmake -DROCM_DEP_ROCMCORE=ON "$COMPONENT_SRC"
    make -j8
    make install
    make package

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cpack -G ${PKGTYPE^^}
    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR
    cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}
clean_rocjpeg() {
    echo "Cleaning rocJPEG build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}
stage2_command_args "$@"
case $TARGET in
    build) build_rocjpeg ;;
    outdir) print_output_directory ;;
    clean) clean_rocjpeg ;;
    *) die "Invalid target $TARGET" ;;
esac
