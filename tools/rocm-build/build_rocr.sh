#!/bin/bash


source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"
PROJ_NAME="rocr"

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...] [make options]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of type referred to by pkg_type"
    echo "  -h,  --help               Prints this help"
    echo "  -s,  --static             Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -n,  --norocr             Don't build ROCr runtime (default is to build). This implies --norocrtst."
    echo "  -k,  --nokfdtest          Don't build kfdtest (default is to build)"
    echo "  -w,  --wheel              Creates python wheel packages. It needs to be used along with -r option"
    echo "  -t,  --norocrtst          Don't build rocrtst (default is to build)"
    echo ""
    echo "  rocrtst options:"
    echo "  -e,  --emulator           Build a version suitable for running on emulator"
    echo "  -g,  --gpu_list <gpus>    Quoted, semi-colon separated list of gpu architectures that"
    echo "                            kernels will run on; e.g., \"gfx803;gfx900;...\" the"
    echo "                            default is to build kernels for all supported architectures."

    echo
    echo "Default build: debug, shared libs"

    return 0
}

build_rocr_runtime() {
    echo "Build ROCr Runtime"
    echo "$ROCR_ROOT"

    if [ "$shared_libs" == "OFF" ]; then
      install_drmStatic_lib
    fi

    if [ ! -d "$rocr_build_dir" ]; then
        mkdir -p "$rocr_build_dir"
        pushd "$rocr_build_dir" || { echo "Failed to pushd into $rocr_build_dir"; exit 1; }
        print_lib_type "$shared_libs"

        cmake \
            $(rocm_cmake_params) \
            -DBUILD_SHARED_LIBS="$shared_libs" \
            -DBUILD_ROCR="$rocr_target" \
            -DENABLE_LDCONFIG=OFF \
            $(rocm_common_cmake_params) \
            -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
            -DROCM_INSTALL_PATH="$ROCM_INSTALL_PATH" \
            -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
            -DTHUNK_DEFINITIONS="$thunk_defines_string" \
            -DROCR_DEFINITIONS="$rocr_defines_string" \
            "$ROCR_ROOT"
        popd
    fi

    cmake --build "$rocr_build_dir" --verbose  -- $DASH_JAY
    cmake --build "$rocr_build_dir" --target install --verbose
    cmake --build "$rocr_build_dir" --target package --verbose
    mkdir -p "$package_lib"

    copy_if DEB "${CPACKGEN:-"DEB;RPM"}" "$package_root_deb" "$rocr_build_dir"/hsa-rocr*.deb
    copy_if RPM "${CPACKGEN:-"DEB;RPM"}" "$package_root_rpm" "$rocr_build_dir"/hsa-rocr*.rpm
}

build_rocrtst() {
    rocrtst_build_type="debug"
    mkdir -p "$rocrtst_build_dir"
    pushd "$rocrtst_build_dir"  || { echo "Failed to pushd into $rocrtst_build_dir"; exit 1; }

    BUILD_TYPE=
    if [[ $gpu_list ]]; then
        cmake -DTARGET_DEVICES="$gpu_list" \
        -DROCRTST_BLD_TYPE="$rocrtst_build_type" \
        -DBUILD_SHARED_LIBS="$shared_libs" \
        -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PATH;$ROCM_INSTALL_PATH/llvm" \
        -DCMAKE_VERBOSE_MAKEFILE=1 \
        $(rocm_common_cmake_params) \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
        -DCPACK_PACKAGING_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
        -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
        -DROCM_PATCH_VERSION="$ROCM_LIBPATCH_VERSION" \
        -DROCM_DIR="$ROCM_INSTALL_PATH" \
        -DLLVM_DIR="$ROCM_INSTALL_PATH/llvm/bin" \
        -DOPENCL_DIR="$ROCM_INSTALL_PATH" \
        -DEMULATOR_BUILD="$emulator_build" \
        "$rocrtst_src_root"
    else
        $ADDRESS_SANITIZER cmake -DROCRTST_BLD_TYPE="$rocrtst_build_type" \
        -DCMAKE_VERBOSE_MAKEFILE=1 \
        -DBUILD_SHARED_LIBS="$shared_libs" \
        -DCMAKE_PREFIX_PATH="$ROCM_INSTALL_PATH;$ROCM_INSTALL_PATH/llvm" \
        -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
        -DCPACK_PACKAGING_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
        -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
        $(rocm_common_cmake_params) \
        -DROCM_PATCH_VERSION="$ROCM_LIBPATCH_VERSION" \
        -DROCM_DIR="$ROCM_INSTALL_PATH" \
        -DLLVM_DIR="$ROCM_INSTALL_PATH/llvm/bin" \
        -DOPENCL_DIR="$ROCM_INSTALL_PATH" \
        -DEMULATOR_BUILD="$emulator_build" \
        "$rocrtst_src_root"
    fi
    echo "Making rocrtst:"
    echo "MAKEARG=$MAKEARG [eom]"

    cmake --build . -- $DASH_JAY
    cmake --build . -- rocrtst_kernels

    cmake --build . -- package || true
    mkdir -p "$rocrtst_package"

    echo "Copying rocrtst binaries to $rocrtst_package"
    progressCopy "$rocrtst_build_dir" "$rocrtst_package"
    progressCopy "$ROCRTST_ROOT/thirdparty" "$rocrtst_package/thirdparty" || true

    DEB_FILE=(./rocrtst*.deb)
    if [ -e "${DEB_FILE[0]}" ]; then
        mkdir -p "$package_root_deb"
        progressCopy "${DEB_FILE[@]}" "$package_root_deb"
    fi

    RPM_FILE=(./rocrtst*.rpm)
    if [ -e "${RPM_FILE[0]}" ]; then
        mkdir -p "$package_root_rpm"
        progressCopy "${RPM_FILE[@]}" "$package_root_rpm"
    fi

    mkdir -p "$package_utils"
    progressCopy "$SCRIPT_ROOT/run_rocrtst.sh" "$package_utils"
    popd

}

file_exists(){
      set -- $1
      [ -e "$1" ]
}

build_kfdtest() {
    echo "Building kfdtest"

    mkdir -p "$kfdtest_build_dir"
    pushd "$kfdtest_build_dir" || { echo "Failed to pushd into $kfdtest_build_dir"; exit 1; }

    cmake \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_SHARED_LIBS="$shared_libs" \
        -DCMAKE_PREFIX_PATH="${ROCM_INSTALL_PATH}" \
        -DCPACK_PACKAGING_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
            $(rocm_common_cmake_params) \
        -DADDRESS_SANITIZER="$ADDRESS_SANITIZER" \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH="FALSE" \
        -DCPACK_GENERATOR="${CPACKGEN:-"DEB;RPM"}" \
        -DCPACK_RPM_DEBUGINFO_PACKAGE=YES \
        -DCPACK_RPM_PACKAGE_DEBUG=YES \
        -DCMAKE_SKIP_BUILD_RPATH=TRUE \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--enable-new-dtags -Wl,--rpath,$ROCM_RPATH $LDFLAGS" \
        "$kfdtest_src_root"
    cmake --build . -- $DASH_JAY

    cmake --build . -- package || true
    popd

    mkdir -p "$kfdtest_bin"
    progressCopy "$kfdtest_build_dir" "$kfdtest_bin"
    progressCopy "$kfdtest_build_dir/kfdtest.exclude" "$kfdtest_bin"
    progressCopy "$kfdtest_build_dir/run_kfdtest.sh" "$kfdtest_bin"

    mkdir -p "$package_utils"
    progressCopy "$SCRIPT_ROOT/run_kfdtest.sh" "$package_utils"

    if file_exists $kfdtest_build_dir/kfdtest*.deb ; then
        mkdir -p "$package_root_deb"
        cp "$kfdtest_build_dir"/kfdtest*.deb "$package_root_deb"
    fi

    if file_exists "$kfdtest_build_dir"/kfdtest*.rpm ; then
        mkdir -p "$package_root_rpm"
        cp $kfdtest_build_dir/kfdtest*.rpm "$package_root_rpm"
    fi
}

clean_rocr_runtime() {
    echo "Cleaning ROCr Runtime"

    rm -f $package_lib/libhsakmt.so*
    rm -f $package_lib/libhsakmt.a
    rm -f $package_lib/libhsakmt-staticdrm.a
    rm -f $package_include/hsakmt*.h $package_include/linux/kfd_ioctl.h

    rm -rf "${runtime_build_dir}"
    rm -f  "$package_root"/lib/libhsa-runtime*
    rm -rf "$package_root/lib/cmake/hsa-runtime64"
    rm -rf "$package_root/include/hsa"
    rm -rf "$package_root/share/doc/hsa-runtime64"
    rm -f "$package_root_deb"/hsa-rocr*.deb
    rm -f "$package_root_rpm"/hsa-rocr*.rpm
    rm -f "$package_root_rpm"/hsa_rocr*.whl

    rm -rf "$PACKAGE_ROOT/hsa"

    clean_rocrtst
    clean_kfdtest
}

clean_rocrtst() {
    echo "Cleaning rocrtst"
    rm -rf "${rocrtst_package}"
    rm -rf "${rocrtst_build_dir}"
    rm -rf "${package_root_deb}"/rocrtst*.deb
    rm -rf "${package_root_rpm}"/rocrtst*.rpm
}

clean_kfdtest() {
    echo "Cleaning kfdtest"
    rm -rf "$kfdtest_build_dir"
    rm -rf "$kfdtest_bin"
    rm -rf "$package_root_deb"/kfdtest*.deb
    rm -rf "$package_root_rpm"/kfdtest*.rpm
}

print_output_directory() {
    case ${pkgtype} in
        ("deb")
            echo "${package_root_deb}";;
        ("rpm")
            package_rpm="some_value"
            echo "${package_root_rpm}";;
        (*)
            echo "Invalid package type \"${pkgtype}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

target="build"

kfdtest_target="yes"
rocrtst_target="yes"
rocr_target="ON"

package_root="$(getPackageRoot)"
package_root_deb="${package_root}/deb/$PROJ_NAME"
package_root_rpm="${package_root}/rpm/$PROJ_NAME"

package_lib="$(getLibPath)"

package_include="$(getIncludePath)"
runtime_build_dir="$(getBuildPath runtime)"

BUILD_TYPE="Debug"
shared_libs="ON"
clean_or_out=0;
maketarget="deb"
pkgtype="deb"
WHEEL_PACKAGE=false

thunk_defines_string=
roct_build_dir="${runtime_build_dir}/libhsakmt"

rocr_defines_string=
rocr_build_dir="${runtime_build_dir}/$PROJ_NAME"

rocrtst_package="$(getBinPath)/rocrtst_tests"
rocrtst_build_dir="${runtime_build_dir}/rocrtst"
rocrtst_src_root="$ROCRTST_ROOT/suites/test_common"
emulator_build=0

kfdtest_src_root="$ROCR_ROOT/libhsakmt/tests/kfdtest"
kfdtest_bin="$(getBinPath)/kfdtest"
package_utils="$(getUtilsPath)"
kfdtest_build_dir=${runtime_build_dir}/kfdtest

unset HIP_DEVICE_LIB_PATH
unset ROCM_PATH

valid_str=$(getopt -o hcraswnkteg:o: --long help,clean,release,static,wheel,address_sanitizer,norocr,nokfdtest,norocrtst,emulator,gpu_list:,outdir: -- "$@")
eval set -- "$valid_str"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                target="clean" ; ((clean_or_out|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="RelWithDebInfo" ; shift ;;
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                shared_libs="OFF" ; shift ;;
        (-w | --wheel)
                WHEEL_PACKAGE=true ; shift ;;
        (-n | --norocr)
                rocr_target="OFF"
                rocrtst_target="no"; shift ;;
        (-k | --nokfdtest)
                kfdtest_target="no" ; shift ;;
        (-t | --norocrtst)
                rocrtst_target="no" ; shift ;;
        (-e | --emulator )
                emulator_build=1 ; shift ;;
        (-g | --gpu_list )
                gpu_list=$2 ; shift 2;;
        (-o | --outdir)
                target="outdir"; pkgtype=$2 ; OUT_DIR_SPECIFIED=1 ; ((clean_or_out|=2)) ; shift 2 ;;
        --)     shift; break;; # end delimiter
        (*)
                echo " ${BASH_SOURCE}: UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 22;;
    esac
done

ret_conflict=1
check_conflicting_options $clean_or_out $pkgtype $maketarget
if [ $ret_conflict -ge 30 ]; then
   print_vars $API_NAME $target $BUILD_TYPE $shared_libs $clean_or_out $pkgtype $maketarget
   exit $ret_conflict
fi

case $target in
    (clean) clean_rocr_runtime ;;
    (build) build_rocr_runtime;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $target" ;;
esac

checkchild(){
    if wait "$1"; then
        return;
    else
        die "$2 failed with exit code $?"
    fi
}

# if [ "$target" != "clean" ]; then
#     if [ "$rocrtst_target" == "yes" ]; then
#         build_rocrtst &
#     else
#         true & # Dummy build_rocrtst
#     fi
#     rocrtst_pid=$!
#     if [ "$kfdtest_target" == "yes" ]; then
#         build_kfdtest &
#     else
#        true & # Dummy build_kfdtest
#     fi
#     kfdtest_pid=$!
#     checkchild $kfdtest_pid kfdtest
#     checkchild $rocrtst_pid rocrtst
# fi

echo "Operation complete"
