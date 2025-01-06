#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src rocAL

build_rocal() {

    echo "Start build"

    # Enable ASAN
    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
        set_asan_env_vars
        set_address_sanitizer_on
    fi

#    python3 ${COMPONENT_SRC}/rocAL-setup.py
    pushd /tmp
    # PyBind11
    git clone -b v2.11.1  https://github.com/pybind/pybind11
    cd pybind11 && mkdir build && cd build
    cmake -DDOWNLOAD_CATCH=ON -DDOWNLOAD_EIGEN=ON ../
    make -j$(nproc) && sudo make install
    cd ../..
    # Turbo JPEG
    git clone -b 3.0.2 https://github.com/libjpeg-turbo/libjpeg-turbo.git
    cd libjpeg-turbo && mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib -DWITH_JPEG8=TRUE ..
    make -j$(nproc) && sudo make install
    cd ../..
    # RapidJSON
    git clone https://github.com/Tencent/rapidjson.git
    cd rapidjson && mkdir build && cd build
    cmake .. && make -j$(nproc) && sudo make install
    popd

    mkdir -p $BUILD_DIR && cd $BUILD_DIR

    cmake -DAMDRPP_PATH=$ROCM_PATH ${COMPONENT_SRC}
    make -j${PROC}
    cmake --build . --target PyPackageInstall
    sudo make install
    sudo make package
    sudo chown -R $(id -u):$(id -g) ${BUILD_DIR}

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    mkdir -p $PACKAGE_DIR
    cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR
    show_build_cache_stats
}

clean_rocal() {
    echo "Cleaning rocAL build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

stage2_command_args "$@"

case $TARGET in
    build) build_rocal ;;
    outdir) print_output_directory ;;
    clean) clean_rocal ;;
    *) die "Invalid target $TARGET" ;;
esac
