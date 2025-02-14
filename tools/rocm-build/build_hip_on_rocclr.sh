#!/bin/bash

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -h,  --help                 Prints this help"
    echo "  -c,  --clean                Clean output and delete all intermediate work"
    echo "  -r,  --release              Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer    Enable address sanitizer"
    echo "  -s,  --static               Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -o,  --outdir <pkg_type>    Print path of output directory containing packages of type referred to by pkg_type"
    echo "  -t,  --offload-arch=<arch>  Specify arch for catch tests ex: --offload-arch=gfx1030 --offload-arch=gfx1100"
    echo "  -p,  --package <type>       Specify packaging format"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"
MAKEOPTS="$DASH_JAY"
PROJ_NAME="hip-on-rocclr"

BUILD_PATH="$(getBuildPath $PROJ_NAME)"

TARGET="build"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_SRC="$(getSrcPath)"
PACKAGE_DEB="$PACKAGE_ROOT/deb/$PROJ_NAME"
PACKAGE_RPM="$PACKAGE_ROOT/rpm/$PROJ_NAME"
PREFIX_PATH="$PACKAGE_ROOT"
CORE_BUILD_DIR="$(getBuildPath hsa-core)"
ROCclr_BUILD_DIR="$(getBuildPath rocclr)"
HIPCC_BUILD_DIR="$(getBuildPath hipcc)"
CATCH_BUILD_DIR="$(getBuildPath catch)"
CATCH_SRC="$HIP_CATCH_TESTS_ROOT/catch"
SAMPLES_SRC="$HIP_CATCH_TESTS_ROOT/samples"
SAMPLES_BUILD_DIR="$(getBuildPath samples)"
if [ ! -e "$CATCH_SRC/CMakeLists.txt" ]; then
   echo "Using catch source from hip project" >&2
   CATCH_SRC="$HIP_ON_ROCclr_ROOT/tests/catch"
fi

BUILD_TYPE="Debug"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"
OFFLOAD_ARCH=()

DEFAULT_OFFLOAD_ARCH=(gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1031 gfx1033 gfx1034 gfx1035 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201)

VALID_STR=`getopt -o hcrast:o: --long help,clean,release,address_sanitizer,static,offload-arch=:,outdir: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-t | --offload-arch=)
                OFFLOAD_ARCH+=( "$2" ); ((CLEAN_OR_OUT|=2)); shift 2 ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        --)     shift; break;;

        (*)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac

done

if [ ${#OFFLOAD_ARCH[@]} = 0 ] ; then
      OFFLOAD_ARCH=( "${DEFAULT_OFFLOAD_ARCH[@]}" )
else
    echo "Using user defined offload archs ${OFFLOAD_ARCH[@]} for catch tests";
fi
printf -v OFFLOAD_ARCH_STR -- '--offload-arch=%q ' "${OFFLOAD_ARCH[@]}"

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi

clean_hip_on_rocclr() {
    rm -rf "$BUILD_PATH"
    rm -rf "$PACKAGE_DEB"
    rm -rf "$PACKAGE_RPM"
    rm -rf "$OUT_DIR/hip"
}

build_hip_on_rocclr() {
    if [  -e "$CLR_ROOT/CMakeLists.txt" ]; then
        _HIP_CMAKELIST_DIR="$CLR_ROOT"
        _HIP_CMAKELIST_OPT="-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF"
        if [ -e "$HIPOTHER_ROOT/hipnv" ]; then
            _HIP_CMAKELIST_OPT="$_HIP_CMAKELIST_OPT -DHIPNV_DIR=$HIPOTHER_ROOT/hipnv"
        fi
    elif [ ! -e "$HIPAMD_ROOT/CMakeLists.txt" ]; then
        echo "No $HIPAMD_ROOT/CMakeLists.txt file, skipping hip on rocclr" >&2
        echo "No $HIPAMD_ROOT/CMakeLists.txt file, skipping hip on rocclr"
        exit 0
    else
        _HIP_CMAKELIST_DIR="$HIPAMD_ROOT"
        _HIP_CMAKELIST_OPT=""
    fi

    echo "$_HIP_CMAKELIST_DIR"
    mkdir -p "$BUILD_PATH"
    pushd "$BUILD_PATH"

    if [ ! -e Makefile ]; then
        echo "Building HIP-On-ROCclr CMake environment"
        print_lib_type $SHARED_LIBS

        cmake $(rocm_cmake_params) \
            -DBUILD_SHARED_LIBS=$SHARED_LIBS \
            -DHIP_COMPILER=clang \
            -DHIP_PLATFORM=amd \
            -DHIP_COMMON_DIR="$HIP_ON_ROCclr_ROOT" \
            $(rocm_common_cmake_params) \
            -DCMAKE_HIP_ARCHITECTURES=OFF \
            -DHSA_PATH="$ROCM_INSTALL_PATH" \
            -DCMAKE_SKIP_BUILD_RPATH=TRUE \
            -DCPACK_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
            -DROCM_PATH="$ROCM_INSTALL_PATH" \
            -DHIPCC_BIN_DIR="$HIPCC_BUILD_DIR" \
            -DHIP_CATCH_TEST=1 \
            $_HIP_CMAKELIST_OPT \
            "$_HIP_CMAKELIST_DIR"

        echo "CMake complete"
    fi

    echo "Build and Install HIP"
    cmake --build . -- $MAKEOPTS install "VERBOSE=1"

    popd
}

build_catch_tests() {
   WORKSPACE=`pwd`
   echo "Build catch2 tests independently"
   if [ ! -e "$CATCH_SRC/CMakeLists.txt" ]; then
      echo "catch source not found: $CATCH_SRC" >&2
      exit
   fi
   # build catch
   rm -rf "$CATCH_BUILD_DIR"
   mkdir -p "$CATCH_BUILD_DIR"
   pushd "$CATCH_BUILD_DIR"
   export HIP_PATH="$ROCM_INSTALL_PATH"
   export ROCM_PATH="$ROCM_INSTALL_PATH"
   cmake \
         -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
         -DBUILD_SHARED_LIBS=$SHARED_LIBS \
         -DHIP_PLATFORM=amd \
         -DROCM_PATH="$ROCM_INSTALL_PATH" \
         -DOFFLOAD_ARCH_STR="$OFFLOAD_ARCH_STR" \
         $(rocm_cmake_params) \
         $(rocm_common_cmake_params) \
         -DCPACK_RPM_DEBUGINFO_PACKAGE=FALSE \
         -DCPACK_DEBIAN_DEBUGINFO_PACKAGE=FALSE \
         -DCPACK_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
         "$CATCH_SRC"


   make $MAKEOPTS build_tests
   echo "Packaging catch tests"
   make $MAKEOPTS package_test
   copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" *.deb
   copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" *.rpm
   popd
}

package_samples() {
   if [ "$ASAN_CMAKE_PARAMS" == "true" ] ; then
      echo "Disable the packaging of HIP samples" >&2
      return
   fi
   WORKSPACE=`pwd`
   if [ ! -e "$SAMPLES_SRC/CMakeLists.txt" ]; then
      echo "HIP samples source not found at: $SAMPLES_SRC" >&2
      echo "Using samples package from hip project: $BUILD_PATH" >&2
      return
   fi

   rm -rf "$SAMPLES_BUILD_DIR"
   mkdir -p "$SAMPLES_BUILD_DIR"
   pushd "$SAMPLES_BUILD_DIR"
   local CMAKE_PATH="$(getCmakePath)"
   export HIP_PATH="$ROCM_INSTALL_PATH"
   export ROCM_PATH="$ROCM_INSTALL_PATH"
   cmake \
         -DROCM_PATH="$ROCM_INSTALL_PATH" \
         -DBUILD_SHARED_LIBS=$SHARED_LIBS \
         $(rocm_cmake_params) \
         $(rocm_common_cmake_params) \
         -DCMAKE_MODULE_PATH="$CMAKE_PATH/hip" \
         -DCPACK_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
         "$SAMPLES_SRC"
   echo "Packaging hip samples from hip-tests project"
   make $MAKEOPTS package_samples
   copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" *.deb
   copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" *.rpm
   popd
}

clean_hip_tests(){
    rm -rf "$CATCH_BUILD_DIR"
    rm -rf "$PACKAGE_SRC/hip-on-rocclr"
    rm -rf "$PACKAGE_SRC/hipamd"
    rm -rf "$PACKAGE_SRC/rocclr"
    rm -rf "$PACKAGE_SRC/opencl-on-rocclr"
    rm -rf "$PACKAGE_SRC/clr"
    rm -rf "$PACKAGE_SRC/hip-tests"
    rm -rf "$PACKAGE_SRC/hipother"
}
copy_hip_tests() {
    clean_hip_tests

    echo "Copy HIP & ROCclr Source and tests"

    mkdir -p "$PACKAGE_SRC/hip-on-rocclr"
    echo "Copying hip-on-rocclr"
    progressCopy "$HIP_ON_ROCclr_ROOT" "$PACKAGE_SRC/hip-on-rocclr"

    if [ -e "$CLR_ROOT/CMakeLists.txt" ]; then
        mkdir -p "$PACKAGE_SRC/clr"
        echo "Copying clr"
        progressCopy "$CLR_ROOT" "$PACKAGE_SRC/clr"
    else
        mkdir -p "$PACKAGE_SRC/hipamd"
        mkdir -p "$PACKAGE_SRC/rocclr"
        mkdir -p "$PACKAGE_SRC/opencl-on-rocclr"
        echo "Copying hipamd"
        progressCopy "$HIPAMD_ROOT" "$PACKAGE_SRC/hipamd"
        echo "Copying rocclr"
        progressCopy "$ROCclr_ROOT" "$PACKAGE_SRC/rocclr"
        echo "Copying opencl-on-rocclr"
        progressCopy "$OPENCL_ON_ROCclr_ROOT" "$PACKAGE_SRC/opencl-on-rocclr"
    fi

    if [ -e "$HIPOTHER_ROOT/hipnv" ]; then
        mkdir -p "$PACKAGE_SRC/hipother"
        echo "Copying hipother"
        progressCopy "$HIPOTHER_ROOT" "$PACKAGE_SRC/hipother"
    fi

    mkdir -p "$PACKAGE_SRC/hip-tests"
    echo "Copying hip-tests"
    progressCopy "$HIP_CATCH_TESTS_ROOT" "$PACKAGE_SRC/hip-tests"
}

package_hip_on_rocclr()
{
    echo "Packagin HIP-on-ROCclr"
    pushd "$BUILD_PATH"
    cmake --build . -- $MAKEOPTS package
    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_DEB" *.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$PACKAGE_RPM" *.rpm
    popd
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

case $TARGET in
    (clean) clean_hip_on_rocclr; clean_hip_tests ;;
    (build) build_hip_on_rocclr; build_catch_tests; package_hip_on_rocclr; package_samples; copy_hip_tests;;
   (outdir) print_output_directory ;;
        (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
