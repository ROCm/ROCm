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
    echo "  -w,  --wheel              Creates python wheel package of omniperf.
                                      It needs to be used along with -r option"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME="omniperf"
PROJ_NAME="$API_NAME"
LIB_NAME="lib${API_NAME}"
TARGET="build"
MAKETARGET="deb"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_LIB="$(getLibPath)"
BUILD_DIR="$(getBuildPath $API_NAME)"
PACKAGE_DEB="$(getPackageRoot)/deb/$API_NAME"
PACKAGE_RPM="$(getPackageRoot)/rpm/$API_NAME"
ROCM_WHEEL_DIR="${BUILD_DIR}/_wheel"
BUILD_TYPE="Debug"
MAKE_OPTS="$DASH_JAY -C $BUILD_DIR"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"
WHEEL_PACKAGE=false


#parse the arguments
VALID_STR=$(getopt -o hcraso:p:w --long help,clean,release,static,address_sanitizer,outdir:,package:,wheel -- "$@")
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        -h | --help)
                printUsage ; exit 0;;
        -c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        -r | --release)
                BUILD_TYPE="Release" ; shift ;;
        -a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        -s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        -o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        -p | --package)
                MAKETARGET="$2" ; shift 2 ;;
        -w | --wheel)
                WHEEL_PACKAGE=true ; shift ;;
        --)     shift; break;; # end delimiter
        *)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac

done

RET_CONFLICT=1
check_conflicting_options "$CLEAN_OR_OUT" "$PKGTYPE" "$MAKETARGET"
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars "$API_NAME" "$TARGET" "$BUILD_TYPE" "$SHARED_LIBS" "$CLEAN_OR_OUT" "$PKGTYPE" "$MAKETARGET"
   exit $RET_CONFLICT
fi

clean() {
    echo "Cleaning $PROJ_NAME"
    rm -rf "$ROCM_WHEEL_DIR"
    rm -rf "$BUILD_DIR"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$PACKAGE_ROOT/${PROJ_NAME:?}"
    rm -rf "$PACKAGE_LIB/${LIB_NAME:?}"*
}

build() {
    echo "Building $PROJ_NAME"
    if [ "$DISTRO_ID" = centos-7 ]; then
        echo "Skip make and uploading packages for Omniperf on Centos7 distro, due to python dependency"
        exit 0
    fi

    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
        pushd "$BUILD_DIR" || exit

        echo "ROCm CMake Params: $(rocm_cmake_params)"
        echo "ROCm Common CMake Params: $(rocm_common_cmake_params)"

        print_lib_type $SHARED_LIBS
        cmake \
            $(rocm_cmake_params) \
            $(rocm_common_cmake_params) \
            -DCHECK_PYTHON_DEPS=NO \
            -DPYTHON_DEPS=${BUILD_DIR}/python-libs \
            -DMOD_INSTALL_PATH=${BUILD_DIR}/modulefiles \
            "$OMNIPERF_ROOT"
    fi

    make $MAKE_OPTS
    make $MAKE_OPTS install
    make $MAKE_OPTS package

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" "$BUILD_DIR/${API_NAME}"*.rpm
}

create_wheel_package() {
    echo "Creating Omniperf wheel package"

    # Copy the setup.py generator to build folder
    mkdir -p "$ROCM_WHEEL_DIR"
    cp -f "$SCRIPT_ROOT"/generate_setup_py.py "$ROCM_WHEEL_DIR"
    cp -f "$SCRIPT_ROOT"/repackage_wheel.sh "$ROCM_WHEEL_DIR"
    cd "$ROCM_WHEEL_DIR" || exit

    # Currently only supports python3.6
    ./repackage_wheel.sh "$BUILD_DIR"/*.rpm python3.6

    # Copy the wheel created to RPM folder which will be uploaded to artifactory
    copy_if WHL "WHL" "$PACKAGE_RPM" "$ROCM_WHEEL_DIR"/dist/*.whl
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo "${PACKAGE_DEB}";;
        ("rpm")
            echo "${PACKAGE_RPM}";;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

verifyEnvSetup

case "$TARGET" in
    (clean) clean ;;
    (build) build ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

if [[ $WHEEL_PACKAGE == true ]]; then
    echo "Wheel Package build started !!!!"
    create_wheel_package
fi

echo "Operation complete"