#!/bin/bash

set_WORK_ROOT(){
    [ -n "$WORK_ROOT" ] && return 0
    export WORK_ROOT=$PWD
    while :; do
	[ -d "$WORK_ROOT/.repo/manifests" ] && return 0
        WORK_ROOT=$WORK_ROOT/..
	( cd -P "$WORK_ROOT" &&  [ "$PWD" != "/" ] ) || break
    done
    echo "Unable to find a .repo/manifests directory above '$PWD'" >&2
    unset WORK_ROOT		# No point in saying we have one when we don't
    return 1
}
set_WORK_ROOT || exit 2

if [ "$DASH_JAY" == "" ]; then
    if [ -x "$(command -v nproc)" ]; then
        export DASH_JAY="-j $(nproc)"
    else
        export DASH_JAY="-j 4"
    fi
fi

export JOB_NAME=release
export JOB_DESIGNATOR="${JOB_DESIGNATOR-"local."}"
echo "JOB_DESIGNATOR=${JOB_DESIGNATOR}"
export SLES_BUILD_ID_PREFIX
echo "SLES_BUILD_ID_PREFIX=${SLES_BUILD_ID_PREFIX}"

if [ -z "${BUILD_ID}" ]; then
    export BUILD_ID=9999
fi

if [ -n "${JOB_NAME}" ]; then
    export ROCM_BUILD_ID=${JOB_NAME/compute-/}-${BUILD_ID}
fi

source /etc/os-release
#re-export the variables with less generic names
export DISTRO_NAME=$ID
export DISTRO_RELEASE=$VERSION_ID
export DISTRO_ID=$ID-$VERSION_ID

case "${DISTRO_NAME}" in
    ("ubuntu") export CPACKGEN=DEB PACKAGEEXT=deb PKGTYPE=deb ROCM_PKGTYPE=DEB ;;
    ("centos") export CPACKGEN=RPM PACKAGEEXT=rpm PKGTYPE=rpm ;;
    ("sles") export CPACKGEN=RPM PACKAGEEXT=rpm PKGTYPE=rpm ;;
    ("rhel") export CPACKGEN=RPM PACKAGEEXT=rpm PKGTYPE=rpm ;;
esac

# set up package file name variables for CPACK_GENERATOR
# rpm packages name are set with jenkins job designator and build no
# deb package is appendeded with OS version as well
export CPACK_DEBIAN_PACKAGE_RELEASE="${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}~$VERSION_ID"
export CPACK_RPM_PACKAGE_RELEASE="${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}"

OUT_DIR="${OUT_DIR:=$WORK_ROOT/out/$DISTRO_ID/$DISTRO_RELEASE}"
export OUT_DIR
export RT_TMP=$OUT_DIR/tmp/rt

#source transform, for things like ocl_lc
export SRC_TF_ROOT=$OUT_DIR/srctf

# Read ROCm Version and calculate ROCm libpatch version from rocm_version.txt
# Using logic from calculateRocmPatchVersion() in common.gvy
get_rocm_libpatch_version() {
    rocm_version=$1
    if [[ "${rocm_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        libpatch_version=${rocm_version//\./0}
        echo "${libpatch_version}"
    else
        echo "Invalid ROCm Version: ${rocm_version}"
        exit 10
    fi
}

# Read the default ROCm version from rocm_version.txt if the ROCM_VERSION
# variable is either not set, empty or only contains spaces.
if [ -f "${WORK_ROOT}/build/rocm_version.txt" ] && [ -z $ROCM_VERSION ]; then
    ROCM_VERSION="$(cat ${WORK_ROOT}/build/rocm_version.txt)"
fi

: ${ROCM_VERSION:="6.3.0"}
ROCM_LIBPATCH_VERSION=$(get_rocm_libpatch_version $ROCM_VERSION)
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "ROCM_LIBPATCH_VERSION=${ROCM_LIBPATCH_VERSION}"
export ROCM_VERSION ROCM_LIBPATCH_VERSION

export ROCM_INSTALL_PATH="/opt/rocm-${ROCM_VERSION}-${BUILD_ID}"
# check if the job = release
if [[ "${JOB_NAME}" == *rel* ]]; then
    export ROCM_INSTALL_PATH="/opt/rocm-${ROCM_VERSION}"
fi

# Setting the ROCM_INSTALL_PATH id to Last Know Good build ID, PSDB incremental built packages will install into /opt/rocm-<parent build ID>
if [ -n "${LKG_BUILD_ID}" ]; then
    export ROCM_INSTALL_PATH="/opt/rocm-${ROCM_VERSION}-${LKG_BUILD_ID}"
fi

echo "Setting ROCM_INSTALL_PATH=${ROCM_INSTALL_PATH}"

export ROCM_PATH="$ROCM_INSTALL_PATH"
export ROCM_LIBPATH=""
export DEVTOOLSET_LIBPATH="/opt/rh/devtoolset-7/root/usr/lib64;/opt/rh/devtoolset-7/root/usr/lib"

# Source directories
# TODO: We should have autodiscoverable makefiles
export DIST_NO_DEBUG=yes
export OPENCL_MAINLINE=1
export HSA_SOURCE_ROOT=$WORK_ROOT/ROCR-Runtime
export HSA_OPENSOURCE_ROOT=$HSA_SOURCE_ROOT/src
export ROCR_ROOT=$WORK_ROOT/ROCR-Runtime
export ROCRTST_ROOT=$HSA_SOURCE_ROOT/rocrtst
export HSA_CORE_ROOT=$HSA_OPENSOURCE_ROOT
export HSA_IMAGE_ROOT=$HSA_OPENSOURCE_ROOT/hsa-ext-image
export HSA_FINALIZE_ROOT=$HSA_OPENSOURCE_ROOT/hsa-ext-finalize
export HSA_TOOLS_ROOT=$HSA_OPENSOURCE_ROOT/hsa-runtime-tools
export OCL_RT_SRC_TF_ROOT=$SRC_TF_ROOT/ocl_lc
export SCRIPT_ROOT=$WORK_ROOT/build
export THUNK_ROOT=$WORK_ROOT/ROCT-Thunk-Interface

export AQLPROFILE_ROOT=$WORK_ROOT/aqlprofile
export ROCPROFILER_ROOT=$WORK_ROOT/rocprofiler
export ROCTRACER_ROOT=$WORK_ROOT/roctracer
export ROCPROFILER_REGISTER_ROOT=$WORK_ROOT/rocprofiler-register
export ROCPROFILER_SDK_ROOT=$WORK_ROOT/rocprofiler-sdk
export ROCPROFILER_COMPUTE_ROOT=$WORK_ROOT/rocprofiler-compute
export ROCPROFILER_SYSTEMS_ROOT=$WORK_ROOT/rocprofiler-systems
export RDC_ROOT=$WORK_ROOT/rdc
export RDCTST_ROOT=$RDC_ROOT/tests/rdc_tests
export UTILS_ROOT=$WORK_ROOT/rocm-utils
export KFDTEST_ROOT=$THUNK_ROOT/tests/kfdtest
if [ -d "$HSA_OPENSOURCE_ROOT/tests/kfdtest" ]; then
	export KFDTEST_ROOT=$HSA_OPENSOURCE_ROOT/tests/kfdtest
fi
export HIPIFY_ROOT=$WORK_ROOT/HIPIFY
export AMD_SMI_LIB_ROOT=$WORK_ROOT/amdsmi
export ROCM_SMI_LIB_ROOT=$WORK_ROOT/rocm_smi_lib
export RSMITST_ROOT=$ROCM_SMI_LIB_ROOT/tests/rocm_smi_test
export LLVM_PROJECT_ROOT=$WORK_ROOT/llvm-project
export LLVM_ROOT=$LLVM_PROJECT_ROOT/llvm
export CLANG_ROOT=$LLVM_PROJECT_ROOT/clang
export LLD_ROOT=$LLVM_PROJECT_ROOT/lld
export HIPCC_ROOT=$LLVM_PROJECT_ROOT/amd/hipcc
export DEVICELIBS_ROOT=$LLVM_PROJECT_ROOT/amd/device-libs
export ROCM_CORE_ROOT=$WORK_ROOT/rocm-core
export ROCM_CMAKE_ROOT=$WORK_ROOT/rocm-cmake
export ROCM_BANDWIDTH_TEST_ROOT=$WORK_ROOT/rocm_bandwidth_test
export ROCMINFO_ROOT=$WORK_ROOT/rocminfo
export ROCR_DEBUG_AGENT_ROOT=$WORK_ROOT/rocr_debug_agent
export COMGR_ROOT=$LLVM_PROJECT_ROOT/amd/comgr
export COMGR_LIB_PATH=$OUT_DIR/build/amd_comgr
export RCCL_ROOT=$WORK_ROOT/rccl
export ROCM_DBGAPI_ROOT=$WORK_ROOT/ROCdbgapi
export ROCM_GDB_ROOT=$WORK_ROOT/ROCgdb
export HIP_ON_ROCclr_ROOT=$WORK_ROOT/HIP
export HIPAMD_ROOT=$WORK_ROOT/hipamd
export HIP_CATCH_TESTS_ROOT=$WORK_ROOT/hip-tests
export CLR_ROOT=$WORK_ROOT/clr
export AOMP_REPOS=$WORK_ROOT/openmp-extras
export HIPOTHER_ROOT=$WORK_ROOT/hipother

# For libraries $ORIGIN
# For binaries $ORIGIN/../lib
export ROCM_LIB_RPATH='$ORIGIN'
export ROCM_EXE_RPATH='$ORIGIN/../lib'

# For ASAN Libraries since the asan lib path is lib/asan/
export ROCM_ASAN_LIB_RPATH='$ORIGIN:$ORIGIN/..'
export ROCM_ASAN_EXE_RPATH="\$ORIGIN/../lib/asan:\$ORIGIN/../lib"

export PATH=$PATH:$SCRIPT_ROOT

# From setup_env.sh
export LIBS_WORK_DIR=$WORK_ROOT
export BUILD_ARTIFACTS=$OUT_DIR/$PACKAGEEXT

export HIPCC_COMPILE_FLAGS_APPEND="-O3 -Wno-format-nonliteral -parallel-jobs=4"
export HIPCC_LINK_FLAGS_APPEND="-O3 -parallel-jobs=4"

export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:${PATH}:${HOME}/.local/bin"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export PROC=${PROC:-"$(nproc)"}
export RELEASE_FLAG=${RELEASE_FLAG:-"-r"}
export SUDO=sudo
export PATH=/usr/local/bin:${PATH}:/sbin:/bin
export CCACHE_DIR=${HOME}/.ccache
