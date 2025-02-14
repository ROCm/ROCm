#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"
PROJ_NAME=OpenCL-ICD-Loader
TARGET="build"
MAKEOPTS="$DASH_JAY"
BUILD_TYPE="Debug"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_DEB="$PACKAGE_ROOT/deb/${PROJ_NAME,,}"
PACKAGE_RPM="$PACKAGE_ROOT/rpm/${PROJ_NAME,,}"
CLEAN_OR_OUT=0;
PKGTYPE="deb"
MAKETARGET="deb"
API_NAME="rocm-opencl-icd-loader"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -h,  --help               Prints this help"
    echo "  -o,  --outdir             Print path of output directory containing packages"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo
    return 0
}

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $TARGET $BUILD_TYPE $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi

clean_opencl_icd_loader() {
    echo "Cleaning $PROJ_NAME"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$PACKAGE_ROOT/${PROJ_NAME,,}"
}

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
    cp -r ./pkg/*/rocm*/* "${ROCM_PATH}" || exit 2
    rm -rf pkg/
}

build_opencl_icd_loader() {
    echo "Downloading $PROJ_NAME" package
    if [ "$DISTRO_NAME" = ubuntu ]; then
        mkdir -p "$PACKAGE_DEB"
        local rocm_ver=${ROCM_VERSION}
        if [ ${ROCM_VERSION##*.} = 0 ]; then
            rocm_ver=${ROCM_VERSION%.*}
        fi
        local url="https://repo.radeon.com/rocm/apt/${rocm_ver}/pool/main/r/${API_NAME}/"
        local package
        package=$(curl -s "$url" | grep -Po 'href="\K[^"]*' | grep "${DISTRO_RELEASE}" | head -n 1)

        if [ -z "$package" ]; then
            echo "No package found for Ubuntu version $DISTRO_RELEASE"
            exit 1
        fi

        wget -t3 -P "$PACKAGE_DEB" "${url}${package}"
        copy_pkg_files_to_rocm ${PROJ_NAME,,} ${API_NAME}
    else
        echo "$DISTRO_ID is not supported..."
        exit 2
    fi

    echo "Installing $PROJ_NAME" package
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

VALID_STR=`getopt -o hcraswlo:p: --long help,clean,release,outdir:,package: -- "$@"`
eval set -- "$VALID_STR"
while true ;
do
    case "$1" in
        (-c  | --clean )
            TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r  | --release )
            BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-h  | --help )
            printUsage ; exit 0 ;;
        (-a  | --address_sanitizer)
            ack_and_ignore_asan ; shift ;;
        (-o  | --outdir)
            TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)
            MAKETARGET="$2" ; shift 2;;
	(-s | --static)
            echo "-s parameter accepted but ignored" ; shift ;;
        --)     shift; break;;
        (*)
            echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac
done

case $TARGET in
    (clean) clean_opencl_icd_loader ;;
    (build) build_opencl_icd_loader ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
