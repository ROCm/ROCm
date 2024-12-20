#!/bin/bash

source "$(dirname "${BASH_SOURCE}")/compute_utils.sh"
export JOB_DESIGNATOR
export BUILD_ID
export SLES_BUILD_ID_PREFIX
export ROCM_LIBPATCH_VERSION

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean                 Clean output and delete all intermediate work"
    echo "  -d,  --debug                 Build a debug version of llvm (excludes packaging)"
    echo "  -r,  --release               Build a release version of the package"
    echo "  -a,  --address_sanitizer     Enable address sanitizer (enabled by default)"
    echo "  -A   --no_address_sanitizer  Disable address sanitizer"
    echo "  -s,  --static                Build static lib (.a).  build instead of dynamic/shared(.so) "
    echo "  -w,  --wheel                 Creates python wheel package of rocm-llvm. It needs to be used along with -r option"
    echo "  -l,  --build_llvm_static     Build LLVM libraries statically linked.  Default is to build dynamic linked libs"
    echo "  -o,  --outdir <pkg_type>     Print path of output directory containing packages of
                            type referred to by pkg_type"
    echo "  -B,  --build                 Build and install binary files into /opt/rocm folder"
    echo "  -P,  --package               Generate packages"
    echo "  -N,  --skip_lit_tests        Skip llvm lit testing (llvm lit testing is enabled by default)"
    echo "  -M,  --skip_man_pages        Skip llvm documentation generation (man pages and rocm-llvm-docs package generation is enabled by default)"
    echo "  -h,  --help                  Prints this help"
    echo
    echo

    return 0
}

PROJ_NAME="lightning"
ROCM_LLVM_LIB_RPATH='\$ORIGIN'
ROCM_LLVM_EXE_RPATH='\$ORIGIN/../lib:\$ORIGIN/../../../lib'

PACKAGE_OUT="$(getPackageRoot)"
BUILD_PATH="$(getBuildPath $PROJ_NAME)"
DEB_PATH="$(getDebPath $PROJ_NAME)"
RPM_PATH="$(getRpmPath $PROJ_NAME)"
INSTALL_PATH="${ROCM_INSTALL_PATH}/lib/llvm"
LLVM_ROOT_LCL="${LLVM_ROOT}"

TARGET="all"
MAKEOPTS="$DASH_JAY"
BUILD_TYPE="Release"
ENABLE_ASSERTIONS=0
SHARED_LIBS="ON"

BUILD_LLVM_DYLIB="OFF"

FLANG_NEW=0
CLEAN_OR_OUT=0;
PKGTYPE="deb"
MAKETARGET="deb"

ASSERT_LLVM_VERSION_MAJOR=""
ASSERT_LLVM_VERSION_MINOR=""

SKIP_LIT_TESTS=0
BUILD_MANPAGES="ON"
STATIC_FLAG=

SANITIZER_AMDGPU=1
    HSA_INC_PATH="$WORK_ROOT/ROCR-Runtime/runtime/hsa-runtime/inc/"
COMGR_INC_PATH="$COMGR_ROOT/include"

VALID_STR=`getopt -o hcV:v:draAswlo:BPNM --long help,clean,assert_llvm_ver_major:,assert_llvm_ver_minor:,debug,release,address_sanitizer,no_address_sanitizer,static,build_llvm_static,wheel,build,package,skip_lit_tests,skip_man_pages,outdir: -- "$@"`
eval set -- "$VALID_STR"

set_dwarf_version(){
  case "$DISTRO_ID" in
    (sles*|rhel*)
       SET_DWARF_VERSION_4="-gdwarf-4"
       ;;
    (*)
       SET_DWARF_VERSION_4=""
       ;;
  esac
  export CFLAGS="$CFLAGS $SET_DWARF_VERSION_4  "
  export CXXFLAGS="$CXXFLAGS $SET_DWARF_VERSION_4 "
  export ASMFLAGS="$ASMFLAGS $SET_DWARF_VERSION_4 "
}

while true ;
do
    #echo "processing $1"
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-V | --assert_llvm_ver_major)
                ASSERT_LLVM_VERSION_MAJOR=$2 ; shift 2 ;;
        (-v | --assert_llvm_ver_minor)
                ASSERT_LLVM_VERSION_MINOR=$2 ; shift 2 ;;
        (-d | --debug)
                BUILD_TYPE="Debug" ; shift ;;
        (-r | --release)
                BUILD_TYPE="Release" ; shift ;;
        (-a | --address_sanitizer)
                set_dwarf_version
                SANITIZER_AMDGPU=1 ;
                HSA_INC_PATH="$WORK_ROOT/hsa/runtime/opensrc/hsa-runtime/inc" ;
                COMGR_INC_PATH="$COMGR_ROOT/include" ; shift ;;
        (-A | --no_address_sanitizer)
                SANITIZER_AMDGPU=0 ;
                unset HSA_INC_PATH ;
                unset COMGR_INC_PATH ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ;
                STATIC_FLAG="-DBUILD_SHARED_LIBS=$SHARED_LIBS" ; shift ;;
        (-l | --build_llvm_static)
                BUILD_LLVM_DYLIB="OFF"; shift ;;
        (-w | --wheel)
                WHEEL_PACKAGE=true ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-B | --build)
                TARGET="build"; shift ;;
        (-P | --package)
                TARGET="package"; shift ;;
        (-N | --skip_lit_tests)
                SKIP_LIT_TESTS=1; shift ;;
        (-M | --skip_man_pages)
                BUILD_MANPAGES="OFF"; shift ;;
        --)     shift; break;; # end delimiter
        (*)
                echo " This should not happen : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac

done

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi

LLVM_PROJECTS="clang;lld;clang-tools-extra"
ENABLE_RUNTIMES="compiler-rt;libunwind"
BOOTSTRAPPING_BUILD_LIBCXX=0
BUILD_AMDCLANG="ON"

ENABLE_RUNTIMES="$ENABLE_RUNTIMES;libcxx;libcxxabi"
BOOTSTRAPPING_BUILD_LIBCXX=1

clean_lightning() {
    rm -rf "$BUILD_PATH"
    rm -rf "$DEB_PATH"
    rm -rf "$RPM_PATH"
}

setup_llvm_info() {
    set +e
    mkdir -p "$LLVM_ROOT_LCL"
    pushd "$LLVM_ROOT_LCL"
    local LLVM_REMOTE_NAME
    local LLVM_URL_NAME
    local LLVM_BRANCH_NAME
    local LLVM_URL_BRANCH

    if [[ "${JOB_NAME}" == *rel* ]]; then
        LLVM_URL_NAME="https://github.com/RadeonOpenCompute/llvm-project"
        LLVM_BRANCH_NAME="roc-${ROCM_VERSION}"
        LLVM_URL_BRANCH="${LLVM_URL_NAME} ${LLVM_BRANCH_NAME}"
    else
        LLVM_REMOTE_NAME=$(git remote)
        LLVM_URL_NAME=$(git config --get remote."${LLVM_REMOTE_NAME}".url)
        LLVM_BRANCH_NAME=$(repo manifest | sed -n 's/.*path="external\/llvm-project".* upstream="\([^"]*\)".*/\1/p' )
        LLVM_URL_BRANCH="${LLVM_URL_NAME} ${LLVM_BRANCH_NAME}"
    fi

    LLVM_COMMIT_GITDATE=$(git show -s --format=@%ct | xargs | date -f - --utc +%y%U%w)
    LLVM_REPO_INFO="${LLVM_URL_BRANCH} ${LLVM_COMMIT_GITDATE}"

    popd
    set -e
}

LLVM_VERSION_MAJOR=""
LLVM_VERSION_MINOR=""
LLVM_VERSION_PATCH=""
LLVM_VERSION_SUFFIX=""
get_llvm_version() {
    local LLVM_VERSIONS=($(awk '/set\(LLVM_VERSION/ {print substr($2,1,length($2)-1)}' ${LLVM_ROOT_LCL}/../cmake/Modules/LLVMVersion.cmake))
    if [ ${#LLVM_VERSIONS[@]} -eq 0 ]; then
        LLVM_VERSIONS=($(awk '/set\(LLVM_VERSION/ {print substr($2,1,length($2)-1)}' ${LLVM_ROOT_LCL}/CMakeLists.txt))
    fi
    LLVM_VERSION_MAJOR=${LLVM_VERSIONS[0]}
    LLVM_VERSION_MINOR=${LLVM_VERSIONS[1]}
    LLVM_VERSION_PATCH=${LLVM_VERSIONS[2]}
    LLVM_VERSION_SUFFIX=${LLVM_VERSIONS[3]}

    echo "Detected LLVM version from source: ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}${LLVM_VERSION_SUFFIX}"
}

create_compiler_config_files() {
    local llvm_bin_dir="${INSTALL_PATH}/bin"
    local rocm_cfg="rocm.cfg"

    {
        echo "-Wl,--enable-new-dtags"
        echo "--rocm-path='<CFGDIR>/../../..'"
        echo "-frtlib-add-rpath"
    } > "${llvm_bin_dir}/$rocm_cfg"

    local compiler_commands=("clang" "clang++" "clang-cpp" "clang-${LLVM_VERSION_MAJOR}" "clang-cl" "flang" "flang-new")
    for i in "${compiler_commands[@]}"; do
        if [ -f "$llvm_bin_dir/$i" ]; then
            local config_file="${llvm_bin_dir}/${i}.cfg"
            echo "@${rocm_cfg}" > $config_file
        fi
    done
}

contains(){

    local target=$1 element
    shift
    for element ; do
        [ "$target" = "$element" ] && return 0
    done
    return 1
}

build_lightning() {
    setup_llvm_info

    get_llvm_version
    if [ -n "${ASSERT_LLVM_VERSION_MAJOR}" ]; then
        echo "Assert LLVM major version: ${ASSERT_LLVM_VERSION_MAJOR}";
        if [ "${ASSERT_LLVM_VERSION_MAJOR}" != "${LLVM_VERSION_MAJOR}" ]; then
            echo "LLVM major version assertion failed, expected ${ASSERT_LLVM_VERSION_MAJOR} but detected ${LLVM_VERSION_MAJOR}!"
            exit 1;
        fi
    fi
    if [ -n "${ASSERT_LLVM_VERSION_MINOR}" ]; then
        echo "Assert LLVM minor version: ${ASSERT_LLVM_VERSION_MINOR}";
        if [ "${ASSERT_LLVM_VERSION_MINOR}" != "${LLVM_VERSION_MINOR}" ]; then
            echo "LLVM minor version assertion failed, expected ${ASSERT_LLVM_VERSION_MINOR} but detected ${LLVM_VERSION_MINOR}!"
            exit 1;
        fi
    fi

    DISABLE_PIE=0

    mkdir -p "$BUILD_PATH"
    pushd "$BUILD_PATH"

    eval EXTRA_LLVM_CMAKE_PARAMS_ARRAY=($EXTRA_LLVM_CMAKE_PARAMS)

    if [ ! -e Makefile ]; then
        echo "Building LLVM CMake environment"
        LLVM_PROJECTS="$LLVM_PROJECTS;mlir"
        if [ -e "$LLVM_ROOT_LCL/../flang/EnableFlangBuild" ]; then
            FLANG_NEW=1
            LLVM_PROJECTS="$LLVM_PROJECTS;flang"
            ENABLE_RUNTIMES="$ENABLE_RUNTIMES;openmp";
        else
            if [[ "${JOB_NAME}" != *afar* ]] && [ -e "$LLVM_ROOT_LCL/../flang/DoROCmRelease" ]; then
              FLANG_NEW=1
              LLVM_PROJECTS="$LLVM_PROJECTS;flang"
            else
                echo "NOT building project flang"
            fi
        fi
        set -x
        cmake $(rocm_cmake_params) ${GEN_NINJA} \
              ${STATIC_FLAG} \
              ${PYTHON_VERSION_WORKAROUND} \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
              -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
              -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
              -DLLVM_ENABLE_RUNTIMES="$ENABLE_RUNTIMES" \
              -DLIBCXX_ENABLE_SHARED=OFF \
              -DLIBCXX_ENABLE_STATIC=ON \
              -DLIBCXX_INSTALL_LIBRARY=OFF \
              -DLIBCXX_INSTALL_HEADERS=OFF \
              -DLIBCXXABI_ENABLE_SHARED=OFF \
              -DLIBCXXABI_ENABLE_STATIC=ON \
              -DLIBCXXABI_INSTALL_STATIC_LIBRARY=OFF \
              -DLLVM_BUILD_DOCS="$BUILD_MANPAGES" \
              -DLLVM_ENABLE_SPHINX="$BUILD_MANPAGES" \
              -DSPHINX_WARNINGS_AS_ERRORS=OFF \
              -DSPHINX_OUTPUT_MAN="$BUILD_MANPAGES" \
              -DLLVM_ENABLE_ASSERTIONS="$ENABLE_ASSERTIONS" \
              -DLLVM_ENABLE_Z3_SOLVER=OFF \
              -DLLVM_ENABLE_ZLIB=ON \
              -DLLVM_AMDGPU_ALLOW_NPI_TARGETS=ON \
              -DCLANG_REPOSITORY_STRING="$LLVM_REPO_INFO" \
              -DCLANG_DEFAULT_PIE_ON_LINUX="$DISABLE_PIE" \
              -DCLANG_DEFAULT_LINKER=lld \
              -DCLANG_DEFAULT_RTLIB=compiler-rt \
              -DCLANG_DEFAULT_UNWINDLIB=libgcc \
              -DCLANG_ENABLE_AMDCLANG="$BUILD_AMDCLANG" \
              -DSANITIZER_AMDGPU="$SANITIZER_AMDGPU" \
              -DPACKAGE_VENDOR="AMD" \
              -DSANITIZER_HSA_INCLUDE_PATH="$HSA_INC_PATH" \
              -DSANITIZER_COMGR_INCLUDE_PATH="$COMGR_INC_PATH" \
              -DLLVM_BUILD_LLVM_DYLIB="$BUILD_LLVM_DYLIB" \
              -DLLVM_LINK_LLVM_DYLIB="$BUILD_LLVM_DYLIB" \
              -DLLVM_ENABLE_LIBCXX="$BUILD_LLVM_DYLIB" \
              -DCMAKE_SKIP_BUILD_RPATH=TRUE\
              -DCMAKE_SKIP_INSTALL_RPATH=TRUE\
              -DCMAKE_EXE_LINKER_FLAGS=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LLVM_EXE_RPATH \
              -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LLVM_LIB_RPATH \
              -DROCM_LLVM_BACKWARD_COMPAT_LINK="$ROCM_INSTALL_PATH/llvm" \
              -DROCM_LLVM_BACKWARD_COMPAT_LINK_TARGET="./lib/llvm" \
              -DCLANG_LINK_FLANG_LEGACY=ON \
              -DCMAKE_CXX_STANDARD=17 \
              -DFLANG_INCLUDE_DOCS=OFF \
              "${EXTRA_LLVM_CMAKE_PARAMS_ARRAY[@]}" \
              "$LLVM_ROOT_LCL"
        set +x
        echo "CMake complete"
    fi

    echo "Building LLVM"

    if [ $BOOTSTRAPPING_BUILD_LIBCXX -eq 1 ]; then
        cmake --build . -- $MAKEOPTS clang lld compiler-rt
        cmake --build . -- $MAKEOPTS runtimes cxx
    fi

    echo "Workaround for race condition"
    echo "End Workaround for race condition"
    cmake --build . -- $MAKEOPTS

    if [ $SKIP_LIT_TESTS -eq 0 ]; then
        if [ $RHEL_BUILD -eq 1 ]; then
            cmake --build . -- $MAKEOPTS check-lld check-mlir
        elif [ "$DISTRO_NAME" != "sles" ]; then
            cmake --build . -- $MAKEOPTS check-llvm check-clang check-lld check-mlir
        fi
    fi
    cmake --build . -- $MAKEOPTS clang-tidy
    cmake --build . -- $MAKEOPTS install

    popd
}

package_lightning_dynamic(){

    if [ "$BUILD_TYPE" == "Debug" ]; then
        return
    fi

    get_llvm_version
    local llvmParsedVersion="${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}"
    local packageName="rocm-llvm"
    local packageSummary="ROCm compiler"
    local packageSummaryLong="ROCm compiler based on LLVM $llvmParsedVersion"
    local installPath="$ROCM_INSTALL_PATH/lib/llvm/"

    if [ "$BUILD_LLVM_DYLIB" == "ON" ] ; then
      local packageNameCore="rocm-llvm-core"
      local packageSummaryCore="ROCm core compiler dylibs"
      local packageSummaryLongCore="ROCm compiler based on LLVM $llvmParsedVersion"
    fi

    local packageArch="amd64"
    local packageVersion="${llvmParsedVersion}.${LLVM_COMMIT_GITDATE}"
    local packageMaintainer="ROCm Compiler Support <rocm.compiler.support@amd.com>"
    local distBin="$INSTALL_PATH/bin"
    local distInc="$INSTALL_PATH/include"
    local distLib="$INSTALL_PATH/lib"
    local distMan="$INSTALL_PATH/share/man"
    local licenseDir="$ROCM_INSTALL_PATH/share/doc/$packageName"
    local licenseDirCore="$ROCM_INSTALL_PATH/share/doc/$packageNameCore"
    local packageDir="$BUILD_PATH/package"
    local backwardsCompatibleSymlink="$ROCM_INSTALL_PATH/llvm"

    local packageDeb="$packageDir/deb"
    local controlFile="$packageDeb/DEBIAN/control"
    local postinstFile="$packageDeb/DEBIAN/postinst"
    local prermFile="$packageDeb/DEBIAN/prerm"
    local specFile="$packageDir/$packageName.spec"
    local debDependencies="python3, libc6, libstdc++6|libstdc++8, libstdc++-5-dev|libstdc++-7-dev|libstdc++-11-dev, libgcc-5-dev|libgcc-7-dev|libgcc-11-dev, rocm-core"
    local debRecommends="gcc, g++, gcc-multilib, g++-multilib"

    local packageRpm="$packageDir/rpm"
    local packageRpmCore="$packageDir/rpm"
    local specFile="$packageDir/$packageName.spec"
    local specFileCore="$packageDir/$packageNameCore.spec"
    local rpmRequires="rocm-core"
    if [ "$BUILD_LLVM_DYLIB" == "ON" ] ; then
        rpmRequires+=", rocm-llvm-core"
    fi
    local rpmRequiresCore="rocm-core"
    local rpmRecommends="gcc, gcc-c++, devtoolset-7-gcc-c++"

    rm -rf "$packageDir"
    echo "rm -rf $packageDir"
    rm -rf "$DEB_PATH"

    local amd_compiler_commands=("amdclang" "amdclang++" "amdclang-cl" "amdclang-cpp" "amdflang" "amdlld" "offload-arch")
    local amd_man_pages=("amdclang.1.gz" "flang.1.gz" "amdflang.1.gz")
    local man_pages=("bugpoint.1" "FileCheck.1" "llvm-ar.1" "llvm-cxxmap.1" "llvm-extract.1" "llvm-lipo.1" "llvm-otool.1" "llvm-readobj.1" "llvm-symbolizer.1" "tblgen.1"
                     "clang.1" "lit.1" "llvm-as.1" "llvm-diff.1" "llvm-ifs.1" "llvm-locstats.1" "llvm-pdbutil.1" "llvm-remarkutil.1" "llvm-tblgen.1" "clang-tblgen.1"
                     "llc.1" "llvm-bcanalyzer.1" "llvm-dis.1" "llvm-install-name-tool.1" "llvm-mca.1" "llvm-profdata.1" "llvm-size.1" "llvm-tli-checker.1" "diagtool.1"
                     "lldb-tblgen.1" "llvm-config.1" "llvm-dwarfdump.1" "llvm-lib.1" "llvm-nm.1" "llvm-profgen.1" "llvm-stress.1" "mlir-tblgen.1" "dsymutil.1" "lli.1"
                     "llvm-cov.1" "llvm-dwarfutil.1" "llvm-libtool-darwin.1" "llvm-objcopy.1" "llvm-ranlib.1" "llvm-strings.1" "opt.1" "extraclangtools.1" "llvm-addr2line.1"
                     "llvm-cxxfilt.1" "llvm-exegesis.1" "llvm-link.1" "llvm-objdump.1" "llvm-readelf.1" "llvm-strip.1")

    if [ "$PACKAGEEXT" = "deb" ]; then

        mkdir -p "$packageDeb/$installPath"
        mkdir -p "$(dirname $controlFile)"
        mkdir -p "$DEB_PATH"
        mkdir -p "$packageDeb/$licenseDir"

        if [ "$BUILD_LLVM_DYLIB" == "ON" ] ; then

          mkdir -p "$packageDeb/$licenseDirCore"

          cp -r "$LLVM_ROOT_LCL/LICENSE.TXT" "$packageDeb/$licenseDirCore"

          cp -P "$distLib/libLLVM"*"so"* "$packageDeb/$installPath/"
          cp -P "$distLib/libFortran"*"so"* "$packageDeb/$installPath/"
          cp -P "$distLib/libclang"*"so"* "$packageDeb/$installPath/"

          echo "Package: $packageNameCore"  > $controlFile
          echo "Architecture: $packageArch" >> $controlFile
          echo "Section: devel" >> $controlFile
          echo "Priority: optional" >> $controlFile
          echo "Maintainer: $packageMaintainer" >> $controlFile
          echo "Version: ${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
          echo "Release:    ${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
          echo "Depends: $debDependencies" >> $controlFile
          echo "Recommends: $debRecommends" >> $controlFile
          echo "Description: $packageSummaryCore" >> $controlFile
          echo "  $packageSummaryLongCore" >> $controlFile

          fakeroot dpkg-deb -Zgzip --build $packageDeb \
          "${DEB_PATH}/${packageNameCore}_${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}_${packageArch}.deb"

          rm -rf "$controlFile"
          rm -rf "$packageDeb/$licenseDirCore"

          rm -f "$packageDeb/$installPath/libLLVM"*"so"*
          rm -f "$packageDeb/$installPath/libFortran"*"so"*
          rm -f "$packageDeb/$installPath/libclang"*"so"*

          mkdir -p "$(dirname $controlFile)"

          rm -rf "$packageDeb/$installPath/*"

          debDependencies="${debDependencies}, ${packageNameCore}"
        fi

        cp -r "$LLVM_ROOT_LCL/LICENSE.TXT" "$packageDeb/$licenseDir"
        cp -r "$distBin" "$packageDeb/$installPath/bin"
        cp -r "$distInc" "$packageDeb/$installPath/include"
        cp -r "$distLib" "$packageDeb/$installPath/lib"
        if [ "$BUILD_MANPAGES" == "ON" ]; then
            for i in "${man_pages[@]}"; do
                gzip -f "$distMan/man1/$i"
            done
            if [ -f "$distMan/man1/clang.1.gz" ]; then
                for i in "${amd_man_pages[@]}"; do
                    ln -sf "clang.1.gz" "$distMan/man1/$i"
                done
            fi
        fi
        cp -r "$distMan" "$packageDeb/$installPath/share"

        touch "$postinstFile" "$prermFile"
        echo "mkdir -p \"$ROCM_INSTALL_PATH/bin\"" >> $postinstFile
        for i in "${amd_compiler_commands[@]}"; do
            if [ -f "$packageDeb/$installPath/bin/$i" ]; then
                echo "ln -s \"../lib/llvm/bin/$i\" \"$ROCM_INSTALL_PATH/bin/$i\"" >> $postinstFile
                echo "rm -f \"$ROCM_INSTALL_PATH/bin/$i\"" >> $prermFile
            fi
        done
        echo "rmdir --ignore-fail-on-non-empty \"$ROCM_INSTALL_PATH/bin\"" >> $prermFile
        chmod 0555 "$postinstFile" "$prermFile"
        cp -P "$backwardsCompatibleSymlink" "$packageDeb/$ROCM_INSTALL_PATH"

        echo "Package: $packageName"  > $controlFile
        echo "Architecture: $packageArch" >> $controlFile
        echo "Section: devel" >> $controlFile
        echo "Priority: optional" >> $controlFile
        echo "Maintainer: $packageMaintainer" >> $controlFile
        echo "Version: ${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
        echo "Release:    ${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
        echo "Depends: $debDependencies" >> $controlFile
        echo "Recommends: $debRecommends" >> $controlFile
        echo "Description: $packageSummary" >> $controlFile
        echo "  $packageSummaryLong" >> $controlFile

        fakeroot dpkg-deb -Zgzip --build $packageDeb \
        "${DEB_PATH}/${packageName}_${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}_${packageArch}.deb"
    fi

    if [ "$PACKAGEEXT" = "rpm" ]; then
        mkdir -p "$(dirname $specFile)"
        rm -rf "$RPM_PATH"
        mkdir -p "$RPM_PATH"

        if [ "$BUILD_LLVM_DYLIB" == "ON" ] ; then
          echo "Name:       $packageNameCore" > $specFileCore
          echo "Version:    ${packageVersion}.${ROCM_LIBPATCH_VERSION}" >> $specFileCore
          echo "Release:    ${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}%{?dist}" >> $specFileCore
          echo "Summary:    $packageSummaryCore" >> $specFileCore
          echo "Group:      System Environment/Libraries" >> $specFileCore
          echo "License:    ASL 2.0 with exceptions" >> $specFileCore
          echo "Requires:   $rpmRequiresCore" >> $specFileCore

          echo "%description" >> $specFileCore
          echo "$packageSummaryLongCore" >> $specFileCore

          echo "%prep" >> $specFileCore
          echo "%setup -T -D -c -n $packageNameCore" >> $specFileCore

          echo "%install" >> $specFileCore
          echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFileCore
          echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/lib" >> $specFileCore
          echo "mkdir -p  \$RPM_BUILD_ROOT/$licenseDirCore" >> $specFileCore

          echo "cp -R $LLVM_ROOT_LCL/LICENSE.TXT \$RPM_BUILD_ROOT/$licenseDirCore" >> $specFileCore

          echo "cp -RP $distLib/libLLVM*so* \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFileCore
          echo "cp -RP $distLib/libFortran*so* \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFileCore
          echo "cp -RP $distLib/libclang*so* \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFileCore

          echo "%clean" >> $specFileCore
          echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFileCore
          echo "%files " >> $specFileCore
          echo "%defattr(-,root,root,-)" >> $specFileCore
          echo "$ROCM_INSTALL_PATH" >> $specFileCore

          echo "%post" >> $specFileCore
          echo "%preun" >> $specFileCore
          echo "%postun" >> $specFileCore

          echo "rpmbuild --define _topdir $packageRpmCore -ba $specFileCore"
          rpmbuild --define "_topdir $packageRpmCore" -ba $specFileCore

          mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH
        fi

        echo "Name:       $packageName" > $specFile
        echo "Version:    ${packageVersion}.${ROCM_LIBPATCH_VERSION}" >> $specFile
        echo "Release:    ${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}%{?dist}" >> $specFile
        echo "Summary:    $packageSummary" >> $specFile
        echo "Group:      System Environment/Libraries" >> $specFile
        echo "License:    ASL 2.0 with exceptions" >> $specFile
        echo "Requires:   $rpmRequires" >> $specFile
        # The following is commented as Centos 7 has a version of rpm
        # that does not understand it. When we no longer support Centos 7
        # then we should have a correct recommends line.
        #echo "Recommends: $rpmRecommends" >> $specFile

        echo "%description" >> $specFile
        echo "$packageSummaryLong" >> $specFile

        echo "%prep" >> $specFile
        echo "%setup -T -D -c -n $packageName" >> $specFile

        echo "%install" >> $specFile
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/bin" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/include" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/lib" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/share/man" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$licenseDir" >> $specFile

        echo "cp -R $LLVM_ROOT_LCL/LICENSE.TXT \$RPM_BUILD_ROOT/$licenseDir" >> $specFile
        echo "cp -P $backwardsCompatibleSymlink \$RPM_BUILD_ROOT/$ROCM_INSTALL_PATH" >> $specFile
        echo "cp -R $distBin \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "cp -R $distInc \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "cp -R $distLib \$RPM_BUILD_ROOT/$installPath" >> $specFile
        if [ "$BUILD_MANPAGES" == "ON" ]; then
            for i in "${man_pages[@]}"; do
                echo "gzip -f $distMan/man1/$i" >> $specFile
            done
            if [ -f "$distMan/man1/clang.1.gz" ]; then
                for i in "${amd_man_pages[@]}"; do
                    echo "ln -sf clang.1.gz \"$distMan/man1/$i\"" >> $specFile
                done
            fi
        fi
        echo "cp -R $distMan \$RPM_BUILD_ROOT/$installPath/share" >> $specFile

        echo "%clean" >> $specFile
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "%files " >> $specFile
        if [ "$BUILD_LLVM_DYLIB" == "ON" ] ; then
          echo "%exclude $installPath/lib/libLLVM*so*" >> $specFile
          echo "%exclude $installPath/lib/libFortran*so*" >> $specFile
          echo "%exclude $installPath/lib/libclang*so*" >> $specFile
        fi

        echo "%defattr(-,root,root,-)" >> $specFile
        echo "$ROCM_INSTALL_PATH" >> $specFile

        echo "%post" >> $specFile
        echo "mkdir -p \"$ROCM_INSTALL_PATH/bin\"" >> $specFile
        for i in "${amd_compiler_commands[@]}"; do
            if [ -f "$distBin/$i" ]; then
                echo "ln -sf ../lib/llvm/bin/$i \"$ROCM_INSTALL_PATH/bin/$i\"" >> $specFile
            fi
        done

        echo "%preun" >> $specFile
        for i in "${amd_compiler_commands[@]}"; do
            if [ -f "$distBin/$i" ]; then
                echo "rm -f \"$ROCM_INSTALL_PATH/bin/$i\"" >> $specFile
            fi
        done
        echo "rmdir --ignore-fail-on-non-empty \"$ROCM_INSTALL_PATH/bin\"" >> $specFile
        echo "%postun" >> $specFile

        rpmbuild --define "_topdir $packageRpm" -ba $specFile
        mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH
    fi
}

package_lightning_static() {

    if [ "$BUILD_TYPE" == "Debug" ]; then
        return
    fi

    get_llvm_version
    local llvmParsedVersion="${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}"

    local packageName="rocm-llvm"
    local packageSummary="ROCm core compiler"
    local packageSummaryLong="ROCm core compiler based on LLVM $llvmParsedVersion"
    if [ "$PACKAGEEXT" = "deb" ]; then
        local packageNameExtra="rocm-llvm-dev"
    else
        local packageNameExtra="rocm-llvm-devel"
    fi
    local packageSummaryExtra="ROCm compiler dev tools"
    local packageSummaryLongExtra="ROCm compiler dev tools and documentation, based on LLVM $llvmParsedVersion"
    local installPath="$ROCM_INSTALL_PATH/lib/llvm/"

    local packageArch="amd64"
    local packageVersion="${llvmParsedVersion}.${LLVM_COMMIT_GITDATE}"
    local packageMaintainer="ROCm Compiler Support <rocm.compiler.support@amd.com>"
    local distBin="$INSTALL_PATH/bin"
    local distInc="$INSTALL_PATH/include"
    local distLib="$INSTALL_PATH/lib"
    local distMan="$INSTALL_PATH/share/man"
    local licenseDir="$ROCM_INSTALL_PATH/share/doc/$packageName"
    local packageDir="$BUILD_PATH/package"
    local backwardsCompatibleSymlink="$ROCM_INSTALL_PATH/llvm"

    local packageDeb="$packageDir/deb"
    local controlFile="$packageDeb/DEBIAN/control"
    local postinstFile="$packageDeb/DEBIAN/postinst"
    local prermFile="$packageDeb/DEBIAN/prerm"
    local specFile="$packageDir/$packageName.spec"
    local debDependencies="python3, libc6, libstdc++6|libstdc++8, libstdc++-5-dev|libstdc++-7-dev|libstdc++-11-dev, libgcc-5-dev|libgcc-7-dev|libgcc-11-dev, rocm-core"
    local debRecommends="gcc, g++, gcc-multilib, g++-multilib"

    local packageRpm="$packageDir/rpm"
    local packageRpmExtra="$packageDir/rpm"
    local specFile="$packageDir/$packageName.spec"
    local specFileExtra="$packageDir/$packageNameExtra.spec"
    local rpmRequires="rocm-core"
    local rpmRequiresExtra="rocm-core, $packageName"
    local rpmRecommends="gcc, gcc-c++, devtoolset-7-gcc-c++"

    rm -rf "$packageDir"
    echo "rm -rf $packageDir"
    rm -rf "$DEB_PATH"

    local amd_compiler_commands=("amdclang" "amdclang++" "amdclang-cl" "amdclang-cpp" "amdflang" "amdlld" "offload-arch")
    local amd_man_pages=("amdclang.1.gz" "flang.1.gz" "amdflang.1.gz")
    local core_bin=("amdgpu-arch" "amdgpu-offload-arch" "amdlld" "amdllvm" "clang" "clang++" "clang-${LLVM_VERSION_MAJOR}" "clang-cl"
                    "clang-cpp" "clang-build-select-link" "clang-offload-bundler" "clang-offload-packager" "clang-offload-wrapper" "clang-linker-wrapper" "clang-nvlink-wrapper" "flang" "flang-new"
                    "ld64.lld" "ld.lld" "llc" "lld" "lld-link" "llvm-ar" "llvm-bitcode-strip" "llvm-dwarfdump" "llvm-install-name-tool"
                    "llvm-link" "llvm-mc" "llvm-objcopy" "llvm-objdump" "llvm-otool" "llvm-ranlib" "llvm-readelf" "llvm-readobj" "llvm-strip"
                    "nvidia-arch" "nvptx-arch" "offload-arch" "opt" "wasm-ld" "amdclang" "amdclang++" "amdclang-${LLVM_VERSION_MAJOR}" "amdclang-cl"
                    "amdclang-cpp" "amdflang"
                    "clang++.cfg" "clang-${LLVM_VERSION_MAJOR}.cfg" "clang-cl.cfg" "clang-cpp.cfg" "clang.cfg" "rocm.cfg")
    local core_lib=("libclang-cpp.so.${LLVM_VERSION_MAJOR}${LLVM_VERSION_SUFFIX}" "libclang-cpp.so" "libclang.so"
                    "libclang.so.${LLVM_VERSION_MAJOR}${LLVM_VERSION_SUFFIX}"
                    "libclang.so.${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}${LLVM_VERSION_SUFFIX}"
                    "libFortranSemantics.a" "libFortranLower.a" "libFortranParser.a" "libFortranCommon.a"
                    "libFortranRuntime.a" "libFortran_main.a" "libFortranDecimal.a" "libFortranEvaluate.a")
    local core_man_pages=("llvm-otool.1" "llvm-readobj.1" "clang.1" "lit.1" "llc.1" "llvm-ar.1" "llvm-dwarfdump.1" "llvm-objcopy.1" "opt.1"
                          "llvm-link.1" "llvm-mc.1" "llvm-objdump.1" "llvm-ranlib.1" "llvm-readelf.1" "llvm-strip.1")
    local dev_man_pages=("bugpoint.1" "FileCheck.1" "llvm-cxxmap.1" "llvm-extract.1" "llvm-lipo.1" "llvm-symbolizer.1"
                           "tblgen.1" "llvm-as.1" "llvm-diff.1" "llvm-ifs.1" "llvm-locstats.1" "llvm-pdbutil.1" "llvm-remarkutil.1"
                           "llvm-tblgen.1" "clang-tblgen.1" "llvm-bcanalyzer.1" "llvm-dis.1" "llvm-install-name-tool.1" "llvm-mca.1"
                           "llvm-profdata.1" "llvm-size.1" "llvm-tli-checker.1" "diagtool.1" "lldb-tblgen.1" "llvm-config.1" "llvm-lib.1"
                           "llvm-nm.1" "llvm-opt-report.1" "llvm-profgen.1" "llvm-reduce.1" "llvm-stress.1" "mlir-tblgen.1" "dsymutil.1"
                           "lli.1" "llvm-cov.1" "llvm-dwarfutil.1" "llvm-libtool-darwin.1" "llvm-strings.1"
                           "extraclangtools.1" "llvm-addr2line.1" "llvm-cxxfilt.1" "llvm-exegesis.1" "scan-build.1")

    if [ "$PACKAGEEXT" = "deb" ]; then
        mkdir -p "$packageDeb/$installPath"
        mkdir "${controlFile%/*}"
        mkdir -p "$DEB_PATH"
        mkdir -p "$packageDeb/$licenseDir"

        cp -r "$LLVM_ROOT_LCL/LICENSE.TXT" "$packageDeb/$licenseDir"

        mkdir -p "$packageDeb/$installPath/bin"
        for i in "${core_bin[@]}"; do
            if [ -f "$distBin/$i" ]; then
                cp -d "$distBin"/$i "$packageDeb/$installPath/bin/"
            fi
        done

        cp -d "$distBin/flang" "$packageDeb/$installPath/bin/"

        mkdir -p "$packageDeb/$installPath/lib/clang"
        cp -r "$distLib/clang/" "$packageDeb/$installPath/lib/"

        if [ $FLANG_NEW -eq 1 ]; then
          mkdir -p "$packageDeb/$installPath/include/flang"
          cp -r "$distInc/flang/" "$packageDeb/$installPath/include/"
        fi

        for i in "${core_lib[@]}"; do
            if [ -f "$distLib/$i" ]; then
                cp -dr "$distLib"/$i "$packageDeb/$installPath/lib"
            fi
        done

        if [ "$BUILD_MANPAGES" == "ON" ]; then
            mkdir -p "$packageDeb/$installPath/share/man1"
            for i in "${core_man_pages[@]}"; do
                if [ -f "$distMan/man1/$i" ]; then
                    gzip -f "$distMan/man1/$i"
                    cp -d "$distMan/man1/${i}.gz" "$packageDeb/$installPath/share/man1/"
                fi
            done
            if [ -f "$distMan/man1/clang.1.gz" ]; then
                for i in "${amd_man_pages[@]}"; do
                    ln -sf "clang.1.gz" "$distMan/man1/$i"
                    cp -d "$distMan/man1/${i}" "$packageDeb/$installPath/share/man1/"
                done
            fi
        fi

        touch "$postinstFile" "$prermFile"
        echo "mkdir -p \"$ROCM_INSTALL_PATH/bin\"" >> $postinstFile
        for i in "${amd_compiler_commands[@]}"; do
            if [ -f "$packageDeb/$installPath/bin/$i" ]; then
                echo "ln -s \"../lib/llvm/bin/$i\" \"$ROCM_INSTALL_PATH/bin/$i\"" >> $postinstFile
                echo "rm -f \"$ROCM_INSTALL_PATH/bin/$i\"" >> $prermFile
            fi
        done
        echo "rmdir --ignore-fail-on-non-empty \"$ROCM_INSTALL_PATH/bin\"" >> $prermFile
        chmod 0555 "$postinstFile" "$prermFile"
        cp -P "$backwardsCompatibleSymlink" "$packageDeb/$ROCM_INSTALL_PATH"

        {
            echo "Package: $packageName"
            echo "Architecture: $packageArch"
            echo "Section: devel"
            echo "Priority: optional"
            echo "Maintainer: $packageMaintainer"
            echo "Version: ${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}"
            echo "Release:    ${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}"
            echo "Depends: $debDependencies"
            echo "Recommends: $debRecommends"
            echo "Description: $packageSummary"
            echo "  $packageSummaryLong"
        } >> "$controlFile"

        fakeroot dpkg-deb -Zgzip --build $packageDeb \
            "${DEB_PATH}/${packageName}_${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}_${packageArch}.deb"

        rm -rf "$controlFile"
        rm -rf "$packageDeb"

        mkdir -p "$packageDeb/$installPath"
        mkdir "${controlFile%/*}"
        mkdir -p "$DEB_PATH"

        mkdir -p "$packageDeb/$installPath/bin"
        for i in "$distBin"/*; do
            bin=$(basename "$i")
            contains "$bin" "${core_bin[@]}" "${amd_compiler_commands[@]}" && continue
            cp -d "$i" "$packageDeb/$installPath/bin/"
        done

        for i in "$distLib"/*; do
            lib=$(basename "$i")
            contains "$lib" "${core_lib[@]}" && continue
            cp -dr "$i" "$packageDeb/$installPath/lib/"
        done
        rm -rf "$packageDeb/$installPath/lib/clang"

        cp -r "$distInc" "$packageDeb/$installPath/include"

        if [ $FLANG_NEW -eq 1 ]; then
          rm -rf "$packageDeb/$installPath/include/flang"
        fi

        if [ "$BUILD_MANPAGES" == "ON" ]; then
            mkdir -p "$packageDeb/$installPath/share/man1"
            for i in "${dev_man_pages[@]}"; do
                if [ -f "$distMan/man1/$i" ]; then
                    gzip -f "$distMan/man1/$i"
                    cp -d "$distMan/man1/${i}.gz" "$packageDeb/$installPath/share/man1/"
                fi
            done
        fi

        debDependencies="${debDependencies}, ${packageName}"

        echo "Package: $packageNameExtra"  > $controlFile
        echo "Architecture: $packageArch" >> $controlFile
        echo "Section: devel" >> $controlFile
        echo "Priority: optional" >> $controlFile
        echo "Maintainer: $packageMaintainer" >> $controlFile
        echo "Version: ${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
        echo "Release:    ${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}" >> $controlFile
        echo "Depends: $debDependencies" >> $controlFile
        echo "Recommends: $debRecommends" >> $controlFile
        echo "Description: $packageSummaryExtra" >> $controlFile
        echo "  $packageSummaryLongExtra" >> $controlFile

        fakeroot dpkg-deb -Zgzip --build $packageDeb \
        "${DEB_PATH}/${packageNameExtra}_${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}_${packageArch}.deb"
    fi

    if [ "$PACKAGEEXT" = "rpm" ]; then

        mkdir -p "$(dirname $specFile)"
        rm -rf "$RPM_PATH"
        mkdir -p "$RPM_PATH"

        echo "Name:       $packageName" > $specFile
        echo "Version:    ${packageVersion}.${ROCM_LIBPATCH_VERSION}" >> $specFile
        echo "Release:    ${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}%{?dist}" >> $specFile
        echo "Summary:    $packageSummary" >> $specFile
        echo "Group:      System Environment/Libraries" >> $specFile
        echo "License:    ASL 2.0 with exceptions" >> $specFile
        echo "Requires:   $rpmRequires" >> $specFile

        echo "%description" >> $specFile
        echo "$packageSummaryLong" >> $specFile

        echo "%prep" >> $specFile
        echo "%setup -T -D -c -n $packageName" >> $specFile

        echo "%install" >> $specFile
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/bin" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$licenseDir" >> $specFile

        echo "cp -R $LLVM_ROOT_LCL/LICENSE.TXT \$RPM_BUILD_ROOT/$licenseDir" >> $specFile
        echo "cp -P $backwardsCompatibleSymlink \$RPM_BUILD_ROOT/$ROCM_INSTALL_PATH" >> $specFile

        for i in "${core_bin[@]}"; do
            if [ -f "$distBin/$i" ]; then
                echo "cp -d \"$distBin\"/$i \$RPM_BUILD_ROOT/$installPath/bin/" >> $specFile
            fi
        done

        echo "cp -d \"$distBin/flang\" \$RPM_BUILD_ROOT/$installPath/bin/" >> $specFile

        echo "cp -d \"$distBin\"/*.cfg \$RPM_BUILD_ROOT/$installPath/bin/" >> $specFile

        echo "mkdir -p \$RPM_BUILD_ROOT/$installPath/lib/clang" >> $specFile
        echo "cp -R \"$distLib/clang/\" \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFile

        if [ $FLANG_NEW -eq 1 ]; then
          echo "mkdir -p \$RPM_BUILD_ROOT/$installPath/include/flang" >> $specFile
          echo "cp -R \"$distInc/flang/\" \$RPM_BUILD_ROOT/$installPath/include/" >> $specFile
        fi

        for i in "${core_lib[@]}"; do
            if [ -f "$distLib/$i" ]; then
                echo "cp -dr \"$distLib\"/$i \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFile
            fi
        done

        if [ "$BUILD_MANPAGES" == "ON" ]; then
            echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/share/man/man1" >> $specFile
            for i in "${core_man_pages[@]}"; do
                if [ -f "$distMan/man1/$i" ]; then
                    echo "gzip -f $distMan/man1/$i" >> $specFile
                    echo "cp -d $distMan/man1/${i}.gz \$RPM_BUILD_ROOT/$installPath/share/man/man1/" >> $specFile
                fi
            done
            if [ -f "$distMan/man1/clang.1.gz" ]; then
                for i in "${amd_man_pages[@]}"; do
                    echo "ln -sf clang.1.gz \"$distMan/man1/$i\"" >> $specFile
                    echo "cp -d $distMan/man1/${i} \$RPM_BUILD_ROOT/$installPath/share/man/man1/" >> $specFile
                done
            fi
        fi

        echo "%clean" >> $specFile
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFile
        echo "%files " >> $specFile
        echo "%defattr(-,root,root,-)" >> $specFile
        {
            echo "$ROCM_INSTALL_PATH"

            echo "%post"
            echo "mkdir -p \"\$RPM_INSTALL_PREFIX0/bin\""
            for i in "${amd_compiler_commands[@]}"; do
                if [ -f "$distBin/$i" ]; then
                    echo "ln -sf ../lib/llvm/bin/$i \"\$RPM_INSTALL_PREFIX0/bin/$i\""
                fi
            done

            echo "%preun"
            for i in "${amd_compiler_commands[@]}"; do
                if [ -f "$distBin/$i" ]; then
                    echo "rm -f \"\$RPM_INSTALL_PREFIX0/bin/$i\""
                fi
            done
            echo "rmdir --ignore-fail-on-non-empty \"\$RPM_INSTALL_PREFIX0/bin\""

            echo "%postun"
        } >> "$specFile"

        echo "rpmbuild --define _topdir $packageRpm -ba $specFile"
        rpmbuild --define "_topdir $packageRpm" -ba $specFile

        mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH

        echo "Name:       $packageNameExtra" > $specFileExtra
        echo "Version:    ${packageVersion}.${ROCM_LIBPATCH_VERSION}" >> $specFileExtra
        echo "Release:    ${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}%{?dist}" >> $specFileExtra
        echo "Summary:    $packageSummaryExtra" >> $specFileExtra
        echo "Group:      System Environment/Libraries" >> $specFileExtra
        echo "License:    ASL 2.0 with exceptions" >> $specFileExtra
        echo "Prefix:     $ROCM_INSTALL_PATH" >> $specFileExtra
        echo "Requires:   $rpmRequiresExtra" >> $specFileExtra
        # The following is commented as Centos 7 has a version of rpm
        # that does not understand it. When we no longer support Centos 7
        # then we should have a correct recommends line.
        #echo "Recommends: $rpmRecommends" >> $specFileExtra

        echo "%description" >> $specFileExtra
        echo "$packageSummaryLongExtra" >> $specFileExtra

        echo "%prep" >> $specFileExtra
        echo "%setup -T -D -c -n $packageNameExtra" >> $specFileExtra

        echo "%install" >> $specFileExtra
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFileExtra
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/bin" >> $specFileExtra
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/include" >> $specFileExtra
        echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/lib" >> $specFileExtra

        echo "cp -P $backwardsCompatibleSymlink \$RPM_BUILD_ROOT/$ROCM_INSTALL_PATH" >> $specFileExtra

        for i in "$distBin"/*; do
            bin=$(basename "$i")
            contains "$bin" "${core_bin[@]}" "${amd_compiler_commands[@]}" && continue
            echo "cp -d \"$i\" \$RPM_BUILD_ROOT/$installPath/bin/" >> $specFileExtra
        done
        for i in "$distLib"/*; do
            lib=$(basename "$i")
            contains "$lib" "${core_lib[@]}" && continue
            echo "cp -dr \"$i\" \$RPM_BUILD_ROOT/$installPath/lib/" >> $specFileExtra
        done

        echo "cp -R $distInc \$RPM_BUILD_ROOT/$installPath" >> $specFileExtra
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath/lib/clang" >> $specFileExtra

        if [ $FLANG_NEW -eq 1 ]; then
          echo "rm -rf \$RPM_BUILD_ROOT/$installPath/include/flang" >> $specFileExtra
        fi

        if [ "$BUILD_MANPAGES" == "ON" ]; then
            echo "mkdir -p  \$RPM_BUILD_ROOT/$installPath/share/man/man1" >> $specFileExtra
            for i in "${dev_man_pages[@]}"; do
                if [ -f "$distMan/man1/$i" ]; then
                    echo "gzip -f $distMan/man1/$i" >> $specFileExtra
                    echo "cp -d \"$distMan/man1/${i}.gz\" \$RPM_BUILD_ROOT/$installPath/share/man/man1/" >> $specFileExtra
                fi
            done
        fi

        echo "%clean" >> $specFileExtra
        echo "rm -rf \$RPM_BUILD_ROOT/$installPath" >> $specFileExtra
        echo "%files " >> $specFileExtra

        echo "%defattr(-,root,root,-)" >> $specFileExtra
        echo "$ROCM_INSTALL_PATH" >> $specFileExtra

        echo "%post" >> $specFileExtra
        echo "%preun" >> $specFileExtra
        echo "%postun" >> $specFileExtra

        rpmbuild --define "_topdir $packageRpmExtra" -ba $specFileExtra
        mv $packageRpmExtra/RPMS/x86_64/*.rpm $RPM_PATH
    fi
}

package_docs() {

    if [ "$BUILD_TYPE" == "Debug"  ]; then
        return
    fi

    if [ "$BUILD_MANPAGES" == "OFF" ]; then
        return
    fi

    get_llvm_version
    local llvmParsedVersion="${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}"

    local packageName="rocm-llvm-docs"
    local packageSummary="ROCm LLVM compiler documentation"
    local packageSummaryLong="Documenation for LLVM $llvmParsedVersion"

    local packageArch="amd64"
    local packageVersion="${llvmParsedVersion}.${LLVM_COMMIT_GITDATE}"
    local packageMaintainer="ROCm Compiler Support <rocm.compiler.support@amd.com>"
    local distDoc="$INSTALL_PATH/share/doc/LLVM"

    local licenseDir="$ROCM_INSTALL_PATH/share/doc/$packageName"
    local packageDir="$BUILD_PATH/package"

    local packageDeb="$packageDir/deb"
    local controlFile="$packageDeb/DEBIAN/control"
    local debDependencies="rocm-core"

    local packageRpm="$packageDir/rpm"
    local specFile="$packageDir/$packageName.spec"
    local rpmRequires="rocm-core"

    rm -rf "$packageDir"
    echo "rm -rf $packageDir"

    if [ "$PACKAGEEXT" = "deb" ]; then

        mkdir -p "$packageDeb/$licenseDir"
        mkdir "${controlFile%/*}"

        cp -r "$LLVM_ROOT_LCL/LICENSE.TXT" "$packageDeb/$licenseDir"

        cp -r "$distDoc" "$packageDeb/$licenseDir"

        {
            echo "Package: $packageName"
            echo "Architecture: $packageArch"
            echo "Section: devel"
            echo "Priority: optional"
            echo "Maintainer: $packageMaintainer"
            echo "Version: ${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}"
            echo "Release:    ${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}"
            echo "Depends: $debDependencies"
            echo "Recommends: $debRecommends"
            echo "Description: $packageSummary"
            echo "  $packageSummaryLong"
        } >> "$controlFile"

        fakeroot dpkg-deb -Zgzip --build $packageDeb \
        "${DEB_PATH}/${packageName}_${packageVersion}.${ROCM_LIBPATCH_VERSION}-${JOB_DESIGNATOR}${BUILD_ID}~${DISTRO_RELEASE}_${packageArch}.deb"
    fi

    if [ "$PACKAGEEXT" = "rpm" ]; then

        mkdir -p "$(dirname $specFile)"
        mkdir -p "$RPM_PATH"

        echo "Name:       $packageName" > $specFile
        echo "Version:    ${packageVersion}.${ROCM_LIBPATCH_VERSION}" >> $specFile
        echo "Release:    ${JOB_DESIGNATOR}${SLES_BUILD_ID_PREFIX}${BUILD_ID}%{?dist}" >> $specFile
        echo "Summary:    $packageSummary" >> $specFile
        echo "Group:      System Environment/Libraries" >> $specFile
        echo "License:    ASL 2.0 with exceptions" >> $specFile
        echo "Requires:   $rpmRequires" >> $specFile

        echo "%description" >> $specFile
        echo "$packageSummaryLong" >> $specFile

        echo "%prep" >> $specFile
        echo "%setup -T -D -c -n $packageName" >> $specFile

        echo "%install" >> $specFile
        echo "mkdir -p  \$RPM_BUILD_ROOT/$licenseDir" >> $specFile

        echo "cp -R $LLVM_ROOT_LCL/LICENSE.TXT \$RPM_BUILD_ROOT/$licenseDir" >> $specFile

        echo "cp -R \"$distDoc\" \$RPM_BUILD_ROOT/$licenseDir" >> $specFile

        echo "%clean" >> $specFile
        echo "%files " >> $specFile
        echo "%defattr(-,root,root,-)" >> $specFile

        echo "$ROCM_INSTALL_PATH" >> $specFile

        rpmbuild --define "_topdir $packageRpm" -ba $specFile
        mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH

    fi

}

print_output_directory() {
    case ${PKGTYPE} in
        ("deb")
            echo ${DEB_PATH};;
        ("rpm")
            echo ${RPM_PATH};;
        (*)
            echo "Invalid package type \"${PKGTYPE}\" provided for -o" >&2; exit 1;;
    esac
    exit
}

build() {
    mkdir -p "${INSTALL_PATH}"
    build_lightning
    create_compiler_config_files
}

create_wheel_package() {
    echo "Creating rocm-llvm wheel package"
    mkdir -p "$ROCM_WHEEL_DIR"
    cp -f $SCRIPT_ROOT/generate_setup_py.py $ROCM_WHEEL_DIR
    cp -f $SCRIPT_ROOT/repackage_wheel.sh $ROCM_WHEEL_DIR
    cd $ROCM_WHEEL_DIR
    # Currently only supports python3.6
    ./repackage_wheel.sh $RPM_PATH/rocm-llvm*.rpm python3.6
    # Copy the wheel created to RPM folder which will be uploaded to artifactory
    mv "$ROCM_WHEEL_DIR"/dist/*.whl "$RPM_PATH"
}

case $TARGET in
    (clean) clean_lightning ;;
    (all)
        time build
        time package_lightning_static
        time package_docs
        ;;
    (build)
        build
        ;;
    (package)
        package_lightning_static
        package_docs
        ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
