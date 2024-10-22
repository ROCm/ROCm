 #!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/envsetup.sh"
source "$(dirname "${BASH_SOURCE[0]}")/compute_utils.sh"

API_NAME=half
HALF_ROOT=$(getPackageRoot $API_NAME)
PACKAGE_SRC=$(pwd)/$API_NAME
PACKAGE_DEB=$(getDebPath $API_NAME)
PACKAGE_RPM=$(getRpmPath $API_NAME)
BUILD_DIR=$(getBuildPath $API_NAME)

TARGET='build'
MAKETARGET='deb'
IGNORE_STATIC="off"

cmdOptionHandler

build_half() {
    echo "Start build"

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
         set_asan_env_vars
         set_address_sanitizer_on
         ASAN_CMAKE_PARAMS="false"
    fi
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

    cmake \
        -DCMAKE_INSTALL_PREFIX="$ROCM_PATH" \
        -DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF \
        "$PACKAGE_SRC"

    cmake --build "$BUILD_DIR" -- $DASH_JAY
    cmake --build "$BUILD_DIR" -- package
    cmake --build "$BUILD_DIR" -- install

    rm -rf _CPack_Packages/ && find -name '*.o' -delete
    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" $BUILD_DIR/half*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" $BUILD_DIR/half*.rpm

}

clean_half() {
    echo "Cleaning half build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}


targetSelector $API_NAME
