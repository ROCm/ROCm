#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -r,  --release            Build a release version of the package"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -w,  --wheel              Creates python wheel package of rocm-cmake. 
                                      It needs to be used along with -r option"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
    type referred to by pkg_type"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -h,  --help               Prints this help"
    echo
    echo

    return 0
}

TARGET="build"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_DEB="$(getPackageRoot)/deb/rocm-cmake"
PACKAGE_RPM="$(getPackageRoot)/rpm/rocm-cmake"
ROCM_CMAKE_BUILD_DIR="$(getBuildPath rocm-cmake)"

ROCM_CMAKE_BUILD_DIR="$(getBuildPath rocm-cmake)"
ROCM_CMAKE_PACKAGE_DEB="$(getPackageRoot)/deb/rocm-cmake"
ROCM_CMAKE_PACKAGE_RPM="$(getPackageRoot)/rpm/rocm-cmake"
ROCM_WHEEL_DIR="${ROCM_CMAKE_BUILD_DIR}/_wheel"
ROCM_CMAKE_BUILD_TYPE="debug"
BUILD_TYPE="Debug"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
PKGTYPE="deb"
MAKETARGET="deb"

VALID_STR=`getopt -o hcraswo:p: --long help,clean,release,static,wheel,address_sanitizer,outdir:,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="Release" ; shift ;;
        (-a | --address_sanitizer)
                ack_and_ignore_asan ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-w | --wheel)
            WHEEL_PACKAGE=true ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)
                MAKETARGET="$2" ; shift 2;;
        --)     shift; break;;
        (*)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac

done

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi


clean_rocm_cmake() {
    rm -rf "$ROCM_WHEEL_DIR"
    rm -rf $ROCM_CMAKE_BUILD_DIR
    rm -rf $ROCM_CMAKE_PACKAGE_DEB
    rm -rf $ROCM_CMAKE_PACKAGE_RPM
}

build_rocm_cmake() {
    echo "Building ROCm cmake"

    if [ ! -d "$ROCM_CMAKE_BUILD_DIR" ]; then
        mkdir -p "$ROCM_CMAKE_BUILD_DIR"
        pushd "$ROCM_CMAKE_BUILD_DIR"

        cmake \
            $(rocm_cmake_params) \
            -DBUILD_SHARED_LIBS=$SHARED_LIBS \
            -DCPACK_SET_DESTDIR="OFF" \
            -DROCM_DISABLE_LDCONFIG=ON \
            $ROCM_CMAKE_ROOT
        cmake --build . -- install
        cmake --build . -- package
        popd
    fi

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$ROCM_CMAKE_PACKAGE_DEB" $ROCM_CMAKE_BUILD_DIR/rocm-cmake*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$ROCM_CMAKE_PACKAGE_RPM" $ROCM_CMAKE_BUILD_DIR/rocm-cmake*.rpm
}

create_wheel_package() {
    echo "Creating rocm-cmake wheel package"
    # Copy the setup.py generator to build folder
    mkdir -p $ROCM_WHEEL_DIR
    cp -f $SCRIPT_ROOT/generate_setup_py.py $ROCM_WHEEL_DIR
    cp -f $SCRIPT_ROOT/repackage_wheel.sh $ROCM_WHEEL_DIR
    cd $ROCM_WHEEL_DIR
    # Currently only supports python3.6
    ./repackage_wheel.sh $ROCM_CMAKE_BUILD_DIR/rocm-cmake*.rpm python3.6
    # Copy the wheel created to RPM folder which will be uploaded to artifactory
    copy_if WHL "WHL" "$ROCM_CMAKE_PACKAGE_RPM" "$ROCM_WHEEL_DIR"/dist/*.whl
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${ROCM_CMAKE_PACKAGE_DEB};;
        ("rpm")
            echo ${ROCM_CMAKE_PACKAGE_RPM};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

case $TARGET in
    (clean) clean_rocm_cmake ;;
    (build) build_rocm_cmake ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

if [[ $WHEEL_PACKAGE == true ]]; then
    echo "Wheel Package build started !!!!"
    create_wheel_package
fi

echo "Operation complete"
