#!/bin/bash

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/compute_helper.sh"

set_component_src hipSPARSELt

while [ "$1" != "" ];
do
    case $1 in
        -o  | --outdir )
            shift 1; PKGTYPE=$1 ; TARGET="outdir" ;;
        -c  | --clean )
            TARGET="clean" ;;
        *)
            break ;;
    esac
    shift 1
done

build_hipsparselt() {
    echo "Start build"

    if [ "${ENABLE_STATIC_BUILDS}" == "true" ]; then
        ack_and_skip_static
    fi

    if [ "${ENABLE_ADDRESS_SANITIZER}" == "true" ]; then
       set_asan_env_vars
       set_address_sanitizer_on
    fi

    cd $COMPONENT_SRC
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    init_rocm_common_cmake_params

    if [ -n "$GPU_ARCHS" ]; then
        GPU_TARGETS="$GPU_ARCHS"
    else
        # gfx940;gfx941;gfx942
        GPU_TARGETS=all
    fi

    FC=gfortran \
    CXX="${ROCM_PATH}/bin/hipcc" \
    cmake \
        -DAMDGPU_TARGETS=${GPU_TARGETS} \
        ${LAUNCHER_FLAGS} \
        "${rocm_math_common_cmake_params[@]}" \
        -DTensile_LOGIC= \
        -DTensile_CODE_OBJECT_VERSION=default \
        -DTensile_CPU_THREADS= \
        -DTensile_LIBRARY_FORMAT=msgpack \
        -DBUILD_CLIENTS_SAMPLES=ON \
        -DBUILD_CLIENTS_TESTS=ON \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} \
        -DBUILD_ADDRESS_SANITIZER="${ADDRESS_SANITIZER}" \
        "$COMPONENT_SRC"

    cmake --build "$BUILD_DIR" -- -j${PROC}
    cmake --build "$BUILD_DIR" -- install
    cmake --build "$BUILD_DIR" -- package

    mkdir -p $PACKAGE_DIR && cp ${BUILD_DIR}/*.${PKGTYPE} $PACKAGE_DIR

    $SCCACHE_BIN -s || echo "Unable to display sccache stats"
}

clean_hipsparselt() {
    echo "Cleaning hipSPARSELt build directory: ${BUILD_DIR} ${PACKAGE_DIR}"
    rm -rf "$BUILD_DIR" "$PACKAGE_DIR"
    echo "Done!"
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${DEB_PATH};;
        ("rpm")
            echo ${RPM_PATH};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

case $TARGET in
    build) build_hipsparselt ;;
    outdir) print_output_directory ;;
    clean) clean_hipsparselt ;;
    *) die "Invalid target $TARGET" ;;
esac
