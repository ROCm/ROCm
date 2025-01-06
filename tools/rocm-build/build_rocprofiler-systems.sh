#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: ${BASH_SOURCE##*/} [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
                                      type referred to by pkg_type"
    echo "  -w,  --wheel              Creates python wheel package of rocprof_sys.
                                      It needs to be used along with -r option"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME="rocprofiler-systems"
PROJ_NAME="$API_NAME"
LIB_NAME="lib${API_NAME}"
TARGET="build"
MAKETARGET="deb"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_LIB="$(getLibPath)"

BUILD_DIR="$(getBuildPath $API_NAME)"
PACKAGE_DEB="$(getPackageRoot)/deb/$API_NAME"
PACKAGE_RPM="$(getPackageRoot)/rpm/$API_NAME"

BUILD_TYPE="Debug"
MAKE_OPTS="-j 8"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0
MAKETARGET="deb"
PKGTYPE="deb"
ASAN=0

VALID_STR=$(getopt -o hcraso:p:w --long help,clean,release,address_sanitizer,static,outdir:,package:,wheel -- "$@")
eval set -- "$VALID_STR"

while true; do
    #echo "parocessing $1"
    case "$1" in
    -h | --help)
        printUsage
        exit 0
        ;;
    -c | --clean)
        TARGET="clean"
        ((CLEAN_OR_OUT |= 1))
        shift
        ;;
    -r | --release)
        BUILD_TYPE="RelWithDebInfo"
        shift
        ;;
    -a | --address_sanitizer)
        ack_and_ignore_asan
        # set_asan_env_vars
        # set_address_sanitizer_on

        ASAN=1
        shift
        ;;
    -s | --static)
        ack_and_skip_static
        ;;
    -o | --outdir)
        TARGET="outdir"
        PKGTYPE=$2
        # OUT_DIR_SPECIFIED=1
        ((CLEAN_OR_OUT |= 2))
        shift 2
        ;;
    -p | --package)
        MAKETARGET="$2"
        shift 2
        ;;
    -w | --wheel)
	   WHEEL_PACKAGE=true
       shift
       ;;
    --)
        shift
        break
        ;; # end delimiter
    *)
        echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] " >&2
        exit 20
        ;;
    esac

done

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
    print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
    exit $RET_CONFLICT
fi

clean() {
    echo "Cleaning $PROJ_NAME"
    rm -rf "$BUILD_DIR"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$PACKAGE_ROOT/${PROJ_NAME:?}"
    rm -rf "$PACKAGE_LIB/${LIB_NAME:?}"*
}

build_rocprofiler_systems() {
    echo "Building $PROJ_NAME"

    if [ $ASAN == 1 ]; then
        echo "Skip make and uploading packages for rocprofiler-systems on ASAN build"
        exit 0
    fi
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        echo "Created build directory: $BUILD_DIR"
    fi

    cd $ROCPROFILER_SYSTEMS_ROOT || exit

    echo "Current submodule status"
    git submodule status
    echo "Cached (old) submodule status"
    git submodule status --cached
    cat .git/config

    echo "Updating submodules"
    git submodule init

    git submodule sync --recursive

    git submodule update --init --recursive --force

    echo "Updated submodule status"
    git submodule status
    cat .git/config

    echo "Build directory: $BUILD_DIR"
    pushd "$BUILD_DIR" || exit
    print_lib_type $SHARED_LIBS

    ELFUTIL_URL="https://compute-artifactory.amd.com/artifactory/rocm-generic-local/dev-tools/omnitrace/elfutils-0.188.tar.bz2"
    BINUTIL_URL="https://compute-artifactory.amd.com/artifactory/rocm-generic-local/dev-tools/omnitrace/binutils-2.40.tar.gz"

    echo "ROCm CMake Params: $(rocm_cmake_params)"
    echo "ROCm Common CMake Params: $(rocm_common_cmake_params)"
    echo "ELFUTIL_URL=$ELFUTIL_URL, BINUTIL_URL=$BINUTIL_URL"

    if [ $ASAN == 1 ]; then
        echo "Address Sanitizer path"

        # Commenting out the below cmake command as it is not working as expected
        # LD_LIBRARY_PATH=$ROCM_INSTALL_PATH/lib/asan:$LD_LIBRARY_PATH
        # cmake \
        #     $(rocm_cmake_params) \
        #     $(rocm_common_cmake_params) \
        #     -DROCPROFSYS_BUILD_{LIBUNWIND,DYNINST}=ON \
        #     -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON \
        #     -DAMDDeviceLibs_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/AMDDeviceLibs" \
        #     -Dhip_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/hip" \
        #     -Dhip-lang_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/hip-lang" \
        #     -Damd_comgr_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/amd_comgr" \
        #     -Dhsa-runtime64_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/hsa-runtime64" \
        #     -Dhsakmt_DIR="${ROCM_INSTALL_PATH}/lib/asan/cmake/hsakmt" \
        #     -DROCM_PATH="${ROCM_INSTALL_PATH}/lib/asan" \
        #     -Drocprofiler_ROOT_DIR="${ROCM_INSTALL_PATH}/lib/asan" \
        #     -DCMAKE_HIP_COMPILER_ROCM_ROOT="${ROCM_INSTALL_PATH}" \
        #     -DCMAKE_PREFIX_PATH="${ROCM_INSTALL_PATH};${ROCM_INSTALL_PATH}/lib/asan" \
        #     -DCMAKE_LIBRARY_PATH="${ROCM_INSTALL_PATH}/lib/asan" \
        #     -DCPACK_DEBIAN_PACKAGE_SHLIBDEPS=OFF \
        #     "$ROCPROFILER_SYSTEMS_ROOT"

    else
        cmake \
            $(rocm_cmake_params) \
            $(rocm_common_cmake_params) \
            -DROCPROFSYS_BUILD_{LIBUNWIND,DYNINST}=ON \
            -DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON \
            -DElfUtils_DOWNLOAD_URL="$ELFUTIL_URL" \
            -D{DYNINST,TIMEMORY}_BINUTILS_DOWNLOAD_URL="$BINUTIL_URL" \
            "$ROCPROFILER_SYSTEMS_ROOT"
    fi


    popd || exit

    echo "Make Options: $MAKE_OPTS"
    cmake --build "$BUILD_DIR" --target all -- $MAKE_OPTS
    cmake --build "$BUILD_DIR" --target install -- $MAKE_OPTS
    cmake --build "$BUILD_DIR" --target package -- $MAKE_OPTS

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" "$BUILD_DIR/${API_NAME}"*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
    "deb")
        echo "${PACKAGE_DEB}"
        ;;
    "rpm")
        echo "${PACKAGE_RPM}"
        ;;
    *)
        echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2
        exit 1
        ;;
    esac
    exit
}

verifyEnvSetup

case "$TARGET" in
    clean) clean ;;
    build) build_rocprofiler_systems ;;
    outdir) print_output_directory ;;
    *) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
