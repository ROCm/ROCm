#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: ${BASH_SOURCE##*/} [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
                                      type referred to by pkg_type"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param & ignore. 
                                      No effect of the param on this build"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

API_NAME="hsa-amd-aqlprofile"
PACKAGE_DEB="$(getPackageRoot)/deb/$API_NAME"
PROJ_NAME="$API_NAME"
TARGET="build"
BUILD_TYPE="Debug"
MAKETARGET="deb"
MAKE_OPTS="$DASH_JAY -C $BUILD_DIR"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0
MAKETARGET="deb"
PKGTYPE="deb"

VALID_STR=$(getopt -o hcro:p: --long help,clean,release,clean,outdir:,package: -- "$@")
eval set -- "$VALID_STR"

while true; do
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
        BUILD_TYPE="Release"
        shift
        ;;
    -o | --outdir)
        TARGET="outdir"
        PKGTYPE=$2
        OUT_DIR_SPECIFIED=1
        ((CLEAN_OR_OUT |= 2))
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] " >&2
        exit 20
        ;;
    esac

done

copy_pkg_files_to_rocm() {
    local comp_folder=$1
    local comp_pkg_name=$2

    cd "${OUT_DIR}/${PKGTYPE}/${comp_folder}"|| exit 2
    if [ "${PKGTYPE}" = 'deb' ]; then
        dpkg-deb -x ${comp_pkg_name}_*.deb pkg/
    else
        mkdir pkg && pushd pkg/ || exit 2
        if [[ "${comp_pkg_name}" != *-dev* ]]; then
            rpm2cpio ../${comp_pkg_name}-*.rpm | cpio -idmv
        else
            rpm2cpio ../${comp_pkg_name}el-*.rpm | cpio -idmv
        fi
        popd || exit 2
    fi
    ls ./pkg -alt
    sudo cp -r ./pkg/*/rocm*/* "${ROCM_PATH}" || exit 2
    rm -rf pkg/
}

clean() {
    echo "Cleaning $PROJ_NAME package"
    rm -rf "$PACKAGE_DEB"
}

build() {
    echo "Downloading $PROJ_NAME" package
    if [ "$DISTRO_NAME" = ubuntu ]; then
        mkdir -p "$PACKAGE_DEB"
        local rocm_ver=${ROCM_VERSION}
        if [ ${ROCM_VERSION##*.} = 0 ]; then
            rocm_ver=${ROCM_VERSION%.*}
        fi
        local url="https://repo.radeon.com/rocm/apt/${rocm_ver}/pool/main/h/${API_NAME}/"
        local package
        package=$(curl -s "$url" | grep -Po 'href="\K[^"]*' | grep "${DISTRO_RELEASE}" | head -n 1)

        if [ -z "$package" ]; then
            echo "No package found for Ubuntu version $DISTRO_RELEASE"
            exit 1
        fi

        wget -t3 -P "$PACKAGE_DEB" "${url}${package}"
        copy_pkg_files_to_rocm ${API_NAME} ${API_NAME}
    else
        echo "$DISTRO_ID is not supported..."
        exit 2
    fi

    echo "Installing $PROJ_NAME" package
}

print_output_directory() {
    case ${PKGTYPE} in
    "deb")
        echo ${PACKAGE_DEB}
        ;;
    "rpm")
        echo ${PACKAGE_RPM}
        ;;
    *)
        echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2
        exit 1
        ;;
    esac
    exit
}

case "$TARGET" in
clean) clean ;;
build) build ;;
outdir) print_output_directory ;;
*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
