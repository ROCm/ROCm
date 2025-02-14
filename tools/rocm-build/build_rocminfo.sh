#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [-c|-r|-h] [makeopts]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Removes all rocminfo build artifacts"
    echo "  -r,  --release            Build non-debug version rocminfo (default is debug)"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -s,  --static             Supports static CI by accepting this param & not bailing out. No effect of the param though"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
                                      type referred to by pkg_type"
    echo "  -h,  --help               Prints this help"
    echo "Possible values for <pkg_type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

PROJ_NAME="rocminfo"
TARGET="build"
ROCMINFO_DEST="$(getBinPath)"
ROCMINFO_SRC_ROOT="$ROCMINFO_ROOT"
ROCMINFO_BUILD_DIR="$(getBuildPath $PROJ_NAME)"

MAKEARG="$DASH_JAY"
PACKAGE_ROOT="$(getPackageRoot)"
ROCMINFO_PACKAGE_DEB="$PACKAGE_ROOT/deb/$PROJ_NAME"
ROCMINFO_PACKAGE_RPM="$PACKAGE_ROOT/rpm/$PROJ_NAME"
BUILD_TYPE="debug"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"


VALID_STR=`getopt -o hcraso:g: --long help,clean,release,static,address_sanitizer,outdir:,gpu_list: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                MAKEARG="$MAKEARG BUILD_TYPE=rel"; BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-g | --gpu_list)
                GPU_LIST="$2" ; shift 2;;
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


clean_rocminfo() {
    echo "Removing rocminfo"
    rm -rf $ROCMINFO_DEST/rocminfo
    rm -rf $ROCMINFO_BUILD_DIR
    rm -rf $ROCMINFO_PACKAGE_DEB
    rm -rf $ROCMINFO_PACKAGE_RPM
}

build_rocminfo() {
    if [ ! -d "$ROCMINFO_BUILD_DIR" ]; then
        mkdir -p $ROCMINFO_BUILD_DIR
        pushd $ROCMINFO_BUILD_DIR

        cmake \
            $(rocm_cmake_params) \
            -DBUILD_SHARED_LIBS=$SHARED_LIBS \
            -DROCRTST_BLD_TYPE="$BUILD_TYPE" \
	    $(rocm_common_cmake_params) \
            -DCPACK_PACKAGE_VERSION_MAJOR="1" \
            -DCPACK_PACKAGE_VERSION_MINOR="$ROCM_LIBPATCH_VERSION" \
            -DCPACK_PACKAGE_VERSION_PATCH="0" \
            -DCMAKE_SKIP_BUILD_RPATH=TRUE\
            $ROCMINFO_SRC_ROOT

        echo "Making rocminfo:"
        cmake --build . -- $MAKEARG
        cmake --build . -- $MAKEARG install
        cmake --build . -- $MAKEARG package
        popd
    fi

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$ROCMINFO_PACKAGE_DEB" $ROCMINFO_BUILD_DIR/*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$ROCMINFO_PACKAGE_RPM" $ROCMINFO_BUILD_DIR/*.rpm
}

print_output_directory() {
     case ${PKGTYPE} in
         ("deb")
             echo ${ROCMINFO_PACKAGE_DEB};;
         ("rpm")
             echo ${ROCMINFO_PACKAGE_RPM};;
         (*)
             echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
     esac
     exit
}

case $TARGET in
    (clean) clean_rocminfo ;;
    (build) build_rocminfo ;;
   (outdir) print_output_directory ;;
        (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
exit 0
