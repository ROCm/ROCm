#!/bin/bash
source "${BASH_SOURCE%/*}/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE[0]}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "       --enable-assertions  Enable assertions"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
            type referred to by pkg_type"
    echo "  -h,  --help               Prints this help"
    echo "  -M,  --skip_man_pages     Do not build the 'docs' target"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME=rocm-dbgapi
AMD_DBGAPI_NAME=amd-dbgapi
MAKEINSTALL_MANIFEST=makeinstall_manifest.txt
PROJ_NAME=$API_NAME
LIB_NAME=lib${API_NAME}.so
TARGET=build
MAKETARGET=deb
PACKAGE_ROOT=$(getPackageRoot)
PACKAGE_LIB=$(getLibPath)
PACKAGE_INCLUDE=$(getIncludePath)
BUILD_DIR=$(getBuildPath $API_NAME)
PACKAGE_DEB=$(getPackageRoot)/deb/$API_NAME
PACKAGE_RPM=$(getPackageRoot)/rpm/$API_NAME
BUILD_TYPE=Debug
MAKE_OPTS=($DASH_JAY -C "$BUILD_DIR")
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"
DODOCSBUILD=true

VALID_STR=$(getopt -o hcraso:p:M --long help,clean,release,enable-assertions,static,address_sanitizer,outdir:,package:skip_man_pages -- "$@")
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ;;
        (-r | --release)
                ENABLE_ASSERTIONS=${ENABLE_ASSERTIONS:-"Off"} ;
                BUILD_TYPE="RelWithDebInfo" ;;
        ( --enable-assertions)
                ENABLE_ASSERTIONS="On" ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ;;
        (-s | --static)
                ack_and_skip_static ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; ((CLEAN_OR_OUT|=2)) ; shift 1 ;;
        (-M | --skip_man_pages) DODOCSBUILD=false;;
        (-p | --package)
                MAKETARGET="$2" ; shift 1;;
        --)     shift; break;;
        (*)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac
    shift

done

check_conflicting_options $CLEAN_OR_OUT "$PKGTYPE" "$MAKETARGET"
if [ "$RET_CONFLICT" -ge 30 ]; then
   print_vars "$API_NAME" "$TARGET" "$BUILD_TYPE" "$SHARED_LIBS" "$CLEAN_OR_OUT" "$PKGTYPE" "$MAKETARGET"
   exit "$RET_CONFLICT"
fi

clean() {
    echo "Cleaning $PROJ_NAME"
    if [ -e "$BUILD_DIR/$MAKEINSTALL_MANIFEST" ] ; then
        xargs rm -f < "$BUILD_DIR/$MAKEINSTALL_MANIFEST"
    fi
    rm -rf "$BUILD_DIR"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "${PACKAGE_ROOT:?}/${PROJ_NAME}"
    rm -rf "${PACKAGE_LIB:?}/${LIB_NAME}"*
    rm -rf "${PACKAGE_LIB:?}/cmake/${AMD_DBGAPI_NAME}"
    rm -rf "${PACKAGE_INCLUDE:?}/${AMD_DBGAPI_NAME}"
}

build() {
    if [ ! -e "$ROCM_DBGAPI_ROOT/CMakeLists.txt" ]
    then
           echo " No $ROCM_DBGAPI_ROOT/CMakeLists.txt file, skipping rocm-dbgapi" >&2
           echo " No $ROCM_DBGAPI_ROOT/CMakeLists.txt file, skipping rocm-dbgapi"
       exit 0
    fi
    echo "Building $PROJ_NAME"

    mkdir -p "$BUILD_DIR"
    pushd "$BUILD_DIR" || exit 99

    cmake \
        $(rocm_cmake_params) \
        $(rocm_common_cmake_params) \
        -DENABLE_ASSERTIONS=${ENABLE_ASSERTIONS:-"On"} \
        "$ROCM_DBGAPI_ROOT"

    popd || exit 99
    cmake --build "$BUILD_DIR" -- "${MAKE_OPTS[@]}"
    "$DODOCSBUILD" && cmake --build "$BUILD_DIR" -- "${MAKE_OPTS[@]}" doc
    cmake --build "$BUILD_DIR" -- "${MAKE_OPTS[@]}" install

    mv "$BUILD_DIR/install_manifest.txt" "$BUILD_DIR/$MAKEINSTALL_MANIFEST"
    cmake --build "$BUILD_DIR" -- "${MAKE_OPTS[@]}" package

    mkdir -p "$PACKAGE_LIB"
    (
        shopt -s nullglob
        cp -R "$BUILD_DIR/lib/${LIB_NAME}"* "$BUILD_DIR/${LIB_NAME}"* "$PACKAGE_LIB"
    )
    
    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" "$BUILD_DIR/${API_NAME}"*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" "$BUILD_DIR/${API_NAME}"*.rpm
}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${PACKAGE_DEB};;
        ("rpm")
            echo ${PACKAGE_RPM};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

verifyEnvSetup

case $TARGET in
    (clean) clean ;;
    (build) build ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
