#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [-c|-r|-h] [makeopts]"
    echo
    echo "Options:"
    echo "  -s,  --static           Component/Build does not support static builds just accepting this param & ignore. No effect of the param on this build"
    echo "  -c,  --clean            Removes all RocR Samples build artifacts"
    echo "  -e,  --emulator         Build a version suitable for running on emulator"
    echo "  -r,  --release          Build release version RocR Samples (default is debug)"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -g,  --gpu_list <gpus>  Semi-colon separated List of gpu architectures that"
    echo "                          kernels will run on; e.g., \"gfx803;gfx900;...\" the"
    echo "                          default is to build kernels for all supported architectures."
    echo "  -h,  --help             Prints this help"
    echo "  makeopts                Options to pass to the make command"
    echo

    return 0
}


GPU_LIST="gfx803;gfx701;gfx801;gfx802;gfx900;gfx902;gfx906;gfx908"

TARGET="build"
ROCRTST_SAMPLES_PACKAGE=$(getBinPath)/rocrtst_samples
ROCRTST_SAMPLES_ROOT=$ROCRTST_ROOT/samples
ROCRTST_SAMPLES_BUILD_DIR=$(getBuildPath rocrtst_samples)

MAKEARG="$DASH_JAY"
PACKAGE_ROOT="$(getPackageRoot)"
PACKAGE_UTILS="$(getUtilsPath)"
ROCRTST_SAMPLES_BUILD_TYPE="debug"
EMULATOR_BUILD=0
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"


#parse the arguments
VALID_STR=`getopt -o hcrao:seg: --long help,clean,release,outdir:,static,address_sanitizer,emulator,gpu_list: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                ROCRTST_SAMPLES_BUILD_TYPE="release"; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-o | --outdir )
                exit ;;
        (-s | --static)
                ack_and_skip_static ;;
        (-e | --emulator )
                EMULATOR_BUILD=1 ; ((CLEAN_OR_OUT|=3)) ; shift ;;
        (-g | --gpu_list )
                GPU_LIST=$2 ; shift 2;;
        --)     shift; break;; # end delimiter
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

clean_rocrsamples() {
    echo "Removing ROCR Samples"
    rm -rf "$ROCRTST_SAMPLES_PACKAGE"
    rm -rf "$ROCRTST_SAMPLES_BUILD_DIR"
}

build_rocrsamples() {
    mkdir -p $ROCRTST_SAMPLES_BUILD_DIR
    pushd $ROCRTST_SAMPLES_BUILD_DIR

    cmake -DTARGET_DEVICES=$GPU_LIST \
        $(rocm_cmake_params) \
        $(rocm_common_cmake_params) \
        -DROCRTST_BLD_TYPE=$ROCRTST_SAMPLES_BUILD_TYPE \
        -DROCM_DIR=$PACKAGE_ROOT \
        -DLLVM_DIR="$ROCM_INSTALL_PATH/llvm/bin" \
        -DOPENCL_DIR=$ROCM_INSTALL_PATH \
        -DEMULATOR_BUILD=$EMULATOR_BUILD \
        $ROCRTST_SAMPLES_ROOT

    echo "Making ROCR Samples:"
    cmake --build . -- $MAKEARG
    cmake --build . -- sample_kernels
    mkdir -p "$ROCRTST_SAMPLES_PACKAGE"

    echo "Copying HSA Sample binaries to $ROCRTST_SAMPLES_PACKAGE"
    progressCopy "$ROCRTST_SAMPLES_BUILD_DIR" "$ROCRTST_SAMPLES_PACKAGE"

    mkdir -p "$PACKAGE_UTILS"
    progressCopy "$SCRIPT_ROOT/run_rocrsamples.sh" "$PACKAGE_UTILS"
    popd
}

case $TARGET in
    clean) clean_rocrsamples ;;
    build) build_rocrsamples ;;
    *) die "Invalid target $target" ;;
esac

echo "Operation complete"
exit 0


