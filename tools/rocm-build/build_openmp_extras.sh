#!/bin/bash

source ${BASH_SOURCE%/*}/compute_utils.sh

printUsage() {
    echo
    echo "Usage: $(basename "${BASH_SOURCE}") [options ...]"
    echo
    echo "Options:"
    echo "  -c,  --clean              Clean output and delete all intermediate work"
    echo "  -p,  --package <type>     Specify packaging format"
    echo "  -r,  --release            Make a release build instead of a debug build"
    echo "  -a,  --address_sanitizer  Enable address sanitizer"
    echo "  -o,  --outdir <pkg_type>  Print path of output directory containing packages of
       type referred to by pkg_type"
    echo "  -s,  --static             Component/Build does not support static builds just accepting this param for configuring package deps"
    echo "  -h,  --help               Prints this help"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

PROJ_NAME="openmp-extras"
packageMajorVersion="18.63"
packageMinorVersion="0"
packageVersion="${packageMajorVersion}.${packageMinorVersion}.${ROCM_LIBPATCH_VERSION}"
BUILD_PATH="$(getBuildPath $PROJ_NAME)"
DEB_PATH="$(getDebPath $PROJ_NAME)"
RPM_PATH="$(getRpmPath $PROJ_NAME)"
TARGET="build"
MAKEOPTS="$DASH_JAY"
STATIC_PKG_DEPS="OFF"

export INSTALL_PREFIX=${ROCM_INSTALL_PATH}

VALID_STR=`getopt -o hcraso:p: --long help,clean,release,address_sanitizer,static,outdir,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        -c  | --clean )
            TARGET="clean" ;;
        -p  | --package )
            TARGET="package" ;;
        -r  | --release )
            ;;
        -a  | --address_sanitizer )
            set_asan_env_vars
            set_address_sanitizer_on
            export ROCM_CMAKECONFIG_PATH="$INSTALL_PREFIX/lib/asan/cmake"
            export VERBOSE=1
            export LDSHARED="$INSTALL_PREFIX/lib/llvm/bin/clang -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2"
            export SANITIZER=1 ;;
        -o  | --outdir )
            shift 1; PKGTYPE=$1 ; TARGET="outdir" ;;
	-s | --static )
            export STATIC_PKG_DEPS="ON" ;;
        -h  | --help )
            printUsage ; exit 0 ;;
        --)     shift; break;;
        *)
            MAKEARG=$@ ; break ;;
    esac
    shift 1
done


clean_openmp_extras() {
    rm -rf "$BUILD_PATH"
    rm -rf "$INSTALL_PREFIX/openmp-extras"
}

toStdoutStderr(){
    printf '%s\n' "$@" >&2
    printf '%s\n' "$@"
}

clean_examples(){
    rm -f "$1"/*.sh
    rm -f "$1"/fortran/*.sh
    rm -f "$1"/openmp/*.sh
}

build_openmp_extras() {
     mkdir -p "$BUILD_PATH"
     pushd "$BUILD_PATH"
     echo "Building openmp-extras"
     echo BUILD_PATH: $BUILD_PATH
     echo "INSTALL_PREFIX:$INSTALL_PREFIX"
     export AOMP_STANDALONE_BUILD=0
     set +e
     checkDevel=$(grep "ENABLE_DEVEL_PACKAGE=ON" $AOMP_REPOS/aomp/bin/build_openmp.sh)
     set -e
     if [ "$checkDevel" == "" ]; then
       export AOMP=$INSTALL_PREFIX/lib/llvm
     else
       export DEVEL_PACKAGE="devel/"
       export AOMP=$INSTALL_PREFIX/openmp-extras
     fi
     export BUILD_AOMP=$BUILD_PATH

     if [ "$EPSDB" == "1" ]; then
       export ROCM_DIR=$ROCM_INSTALL_PATH
     else
       export ROCM_DIR=$INSTALL_PREFIX
     fi

     if [ -d "$ROCM_DIR" ]; then
       echo "--------------------------"
       echo "ROCM_DIR:"
       echo "----------"
       ls $ROCM_DIR
       echo "--------------------------"
     fi
     if [ -d "$ROCM_DIR"/include ]; then
       echo "ROCM_DIR/include:"
       echo "----------"
       ls $ROCM_DIR/include
       echo "--------------------------"
     fi
     if [ -d "$ROCM_DIR"/include/hsa ]; then
       echo "ROCM_DIR/include/hsa:"
       echo "----------"
       ls $ROCM_DIR/include/hsa
       echo "--------------------------"
     fi

     export AOMP_JENKINS_BUILD_LIST="extras openmp pgmath flang flang_runtime"
     echo "BEGIN Build of openmp-extras"
     "$AOMP_REPOS"/aomp/bin/build_aomp.sh $MAKEARG

     local llvm_ver=`$INSTALL_PREFIX/lib/llvm/bin/clang --print-resource-dir | sed 's^/llvm/lib/clang/^ ^' | awk '{print $2}'`
     if [ ! -e $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp.h ] ; then
       if [ ! -h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp.h ] ; then
         ln -s ../../../../include/omp.h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp.h
       fi
     fi
     if [ ! -e $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/ompt.h ] ; then
       if [ ! -h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/ompt.h ] ; then
         ln -s ../../../../include/ompt.h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/ompt.h
       fi
     fi
     if [ ! -e $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp-tools.h ] ; then
       if [ ! -h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp-tools.h ] ; then
         ln -s ../../../../include/omp-tools.h $ROCM_INSTALL_PATH/lib/llvm/lib/clang/$llvm_ver/include/omp-tools.h
       fi
     fi
     popd
}

package_openmp_extras_deb() {
        local packageName=$1
        local packageDeb="$packageDir/deb"
        local packageArch="amd64"
        local packageMaintainer="Openmp Extras Support <openmp-extras.support@amd.com>"
        local packageSummary="OpenMP Extras provides openmp and flang libraries."
        local packageSummaryLong="openmp-extras $packageVersion is based on LLVM 17 and is used for offloading to Radeon GPUs."
        local debDependencies="rocm-llvm, rocm-device-libs, rocm-core"
        local debRecommends="gcc, g++"
        local controlFile="$packageDeb/openmp-extras/DEBIAN/control"

        if [ "$packageName" == "openmp-extras-runtime" ]; then
          packageType="runtime"
          if [ "$STATIC_PKG_DEPS" == "OFF" ]; then
            debDependencies="rocm-core, hsa-rocr"
          else
            echo "static package dependency configuration for runtime" ;
            debDependencies="rocm-core, hsa-rocr-static-dev"
          fi
        else
          local debProvides="openmp-extras"
          local debConflicts="openmp-extras"
          local debReplaces="openmp-extras"
          packageType="devel"
          if [ "$STATIC_PKG_DEPS" == "OFF" ]; then
            debDependencies="$debDependencies, openmp-extras-runtime, hsa-rocr-dev"
          else
            echo "Enabled static package dependency configuration for dev" ;
            debDependencies="$debDependencies, openmp-extras-runtime, hsa-rocr-static-dev"
          fi
        fi

        if [ -f "$BUILD_PATH"/build/installed_files.txt ] && [ ! -d "$INSTALL_PREFIX"/openmp-extras/devel ]; then
          if [ "$packageType" == "runtime" ]; then
            rm -rf "$packageDir"
            rm -rf "$DEB_PATH"
            mkdir -p "$DEB_PATH"
            mkdir -p $packageDeb/openmp-extras

            mkdir -p $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras
            cp -r $AOMP_REPOS/aomp/LICENSE $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.apache2
            cp -r $AOMP_REPOS/aomp-extras/LICENSE $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.mit
            cp -r $AOMP_REPOS/flang/LICENSE.txt $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.flang
	  else
            rm -rf $packageDeb/openmp-extras/*
            mkdir -p $packageDeb/openmp-extras$copyPath/bin
            cp -r --parents "$installPath"/lib-debug/src $packageDeb/openmp-extras
	  fi
        else
          if [ "$packageType" == "runtime" ]; then
            rm -rf "$packageDir"
            rm -rf "$DEB_PATH"
            mkdir -p "$DEB_PATH"
            mkdir -p $packageDeb/openmp-extras$installPath
            mkdir -p $packageDeb/openmp-extras$installPath/lib/clang/$llvm_ver/include
            mkdir -p $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras
            cp -r $AOMP_REPOS/aomp/LICENSE $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.apache2
            cp -r $AOMP_REPOS/aomp-extras/LICENSE $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.mit
            cp -r $AOMP_REPOS/flang/LICENSE.txt $packageDeb/openmp-extras$copyPath/share/doc/openmp-extras/LICENSE.flang
          else
            rm -rf $packageDeb/openmp-extras$installPath/*
            rm -rf $packageDeb/openmp-extras/bin
            rm -rf $packageDeb/openmp-extras$copyPath/share
            echo mkdir -p $packageDeb/openmp-extras$copyPath/bin
            mkdir -p $packageDeb/openmp-extras$copyPath/bin
            mkdir -p $packageDeb/openmp-extras$installPath/lib/clang/$llvm_ver/include
          fi
	fi

        mkdir -p "$(dirname $controlFile)"

	if [ -f "$BUILD_PATH"/build/installed_files.txt ] && [ ! -d "$INSTALL_PREFIX"/openmp-extras/devel ]; then
	  if [ "$packageType" == "runtime" ]; then
	    cat "$BUILD_PATH"/build/installed_files.txt | grep -P '\.so|\.a' | cut -d":" -f2 | cut -d" " -f2 | xargs -I {} cp -d --parents {} "$packageDeb"/openmp-extras

	    cp -d --parents "$installPath/lib/libgomp.so" "$packageDeb"/openmp-extras
	    cp -d --parents "$installPath/lib/libiomp5.so" "$packageDeb"/openmp-extras
	    cp -d --parents "$installPath/lib-debug/libgomp.so" "$packageDeb"/openmp-extras
	    cp -d --parents "$installPath/lib-debug/libiomp5.so" "$packageDeb"/openmp-extras
	  else
	    cat "$BUILD_PATH"/build/installed_files.txt | grep -Pv '\.so|\.a' | cut -d":" -f2 | cut -d" " -f2 | xargs -I {} cp -d --parents {} "$packageDeb"/openmp-extras
	  fi
	else
          cp -r "$AOMP"/"$packageType"/* "$packageDeb"/openmp-extras"$installPath"
	fi

        if [ "$packageType" == "devel" ]; then
          mkdir -p "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          echo cp -r "$AOMP_REPOS"/aomp/examples/fortran "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          cp -r "$AOMP_REPOS"/aomp/examples/fortran "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          cp -r "$AOMP_REPOS"/aomp/examples/openmp "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          cp -r "$AOMP_REPOS"/aomp/examples/tools "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          if [ -e "$AOMP_REPOS/aomp/examples/Makefile.help" ]; then
            cp "$AOMP_REPOS"/aomp/examples/Makefile* "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
          fi
          clean_examples "$packageDeb"/openmp-extras"$copyPath"/share/openmp-extras/examples
        fi

        if [ "$packageType" == "devel" ]; then
          ln -s ../../../../include/omp.h $packageDeb/openmp-extras$installPath/lib/clang/$llvm_ver/include/omp.h
          ln -s ../../../../include/ompt.h $packageDeb/openmp-extras$installPath/lib/clang/$llvm_ver/include/ompt.h
          ln -s ../../../../include/omp-tools.h $packageDeb/openmp-extras$installPath/lib/clang/$llvm_ver/include/omp-tools.h
          if [ ! -h "$packageDeb"/openmp-extras"$copyPath"/bin/aompcc ] && [ -e "$packageDeb"/openmp-extras"$installPath"/bin/aompcc ]; then
            ln -s ../lib/llvm/bin/aompcc "$packageDeb"/openmp-extras"$copyPath"/bin/aompcc
          fi
          if [ -e "$packageDeb"/openmp-extras"$installPath"/bin/mymcpu ]; then
            ln -s ../lib/llvm/bin/mymcpu "$packageDeb"/openmp-extras"$copyPath"/bin/mymcpu
          fi
          if [ -e "$packageDeb"/openmp-extras"$installPath"/bin/mygpu ]; then
            ln -s ../lib/llvm/bin/mygpu "$packageDeb"/openmp-extras"$copyPath"/bin/mygpu
          fi
        fi

        ls -l "$packageDeb"/openmp-extras"$installPath"
	if [ "$packageType" == "devel" ]; then
          ls -l "$packageDeb"/openmp-extras"$installPath"/bin
          ls -l "$packageDeb"/openmp-extras"$copyPath"/bin
	fi

        {
          echo "Package: $packageName"
          echo "Architecture: $packageArch"
          echo "Section: devel"
          echo "Priority: optional"
          echo "Maintainer: $packageMaintainer"
          echo "Version: $packageVersion-${CPACK_DEBIAN_PACKAGE_RELEASE}"
          echo "Depends: $debDependencies"
          echo "Recommends: $debRecommends"
          if [ "$packageType" == "devel" ]; then
            echo "Provides: $debProvides"
            echo "Conflicts: $debConflicts"
            echo "Replaces: $debReplaces"
          fi
          echo "Description: $packageSummary"
          echo "  $packageSummaryLong"
        } > $controlFile
        fakeroot dpkg-deb -Zgzip --build $packageDeb/openmp-extras \
        "$DEB_PATH/${packageName}_${packageVersion}-${CPACK_DEBIAN_PACKAGE_RELEASE}_${packageArch}.deb"
}

package_openmp_extras_asan_deb() {
        local packageName=$1
        local packageDeb="$packageDir/deb"
        local packageArch="amd64"
        local packageMaintainer="Openmp Extras Support <openmp-extras.support@amd.com>"
        local packageSummary="AddressSanitizer OpenMP Extras provides instrumented openmp and flang libraries."
        local packageSummaryLong="openmp-extras $packageVersion is based on LLVM 17 and is used for offloading to Radeon GPUs."
        local debDependencies="hsa-rocr-asan, rocm-core-asan"
        local debRecommends="gcc, g++"
        local controlFile="$packageDeb/openmp-extras/DEBIAN/control"
        local asanLibDir="runtime"

        rm -rf "$packageDir"
        rm -rf "$DEB_PATH"
        mkdir -p "$DEB_PATH"
        local licenseDir="$packageDeb/openmp-extras$copyPath/share/doc/openmp-extras-asan"
        mkdir -p $licenseDir
        cp -r $AOMP_REPOS/aomp/LICENSE $licenseDir/LICENSE.apache2
        cp -r $AOMP_REPOS/aomp-extras/LICENSE $licenseDir/LICENSE.mit
        cp -r $AOMP_REPOS/flang/LICENSE.txt $licenseDir/LICENSE.flang

        mkdir -p "$(dirname $controlFile)"
        if [ -f "$BUILD_PATH"/build/installed_files.txt ] && [ ! -d "$INSTALL_PREFIX"/openmp-extras ]; then
	  cat "$BUILD_PATH"/build/installed_files.txt | grep -P 'asan' | cut -d":" -f2 | cut -d" " -f2 | xargs -I {} cp -d --parents {} "$packageDeb"/openmp-extras

	  cp -d --parents "$installPath/lib/asan/libgomp.so" "$packageDeb"/openmp-extras
	  cp -d --parents "$installPath/lib/asan/libiomp5.so" "$packageDeb"/openmp-extras
	  cp -d --parents "$installPath/lib-debug/asan/libgomp.so" "$packageDeb"/openmp-extras
	  cp -d --parents "$installPath/lib-debug/asan/libiomp5.so" "$packageDeb"/openmp-extras

        else
          mkdir -p $packageDeb/openmp-extras$installPath/lib/asan
          mkdir -p $packageDeb/openmp-extras$installPath/lib-debug/asan
          cp -r "$AOMP"/lib/asan/* "$packageDeb"/openmp-extras"$installPath"/lib/asan/
          cp -r "$AOMP"/lib-debug/asan/* "$packageDeb"/openmp-extras"$installPath"/lib-debug/asan/
          cp -r "$AOMP"/"$asanLibDir"/lib/asan/* "$packageDeb"/openmp-extras"$installPath"/lib/asan/
          cp -r "$AOMP"/"$asanLibDir"/lib-debug/asan/* "$packageDeb"/openmp-extras"$installPath"/lib-debug/asan/
          cp -r "$AOMP"/devel/lib/asan/* "$packageDeb"/openmp-extras"$installPath"/lib/asan/
          cp -r "$AOMP"/devel/lib-debug/asan/* "$packageDeb"/openmp-extras"$installPath"/lib-debug/asan/
        fi

        {
          echo "Package: $packageName"
          echo "Architecture: $packageArch"
          echo "Section: devel"
          echo "Priority: optional"
          echo "Maintainer: $packageMaintainer"
          echo "Version: $packageVersion-${CPACK_DEBIAN_PACKAGE_RELEASE}"
          echo "Depends: $debDependencies"
          echo "Recommends: $debRecommends"
          echo "Description: $packageSummary"
          echo "  $packageSummaryLong"
        } > $controlFile
        fakeroot dpkg-deb -Zgzip --build $packageDeb/openmp-extras \
        "$DEB_PATH/${packageName}_${packageVersion}-${CPACK_DEBIAN_PACKAGE_RELEASE}_${packageArch}.deb"
}


package_openmp_extras_rpm() {
        local packageName=$1
        local packageRpm="$packageDir/rpm"
        local specFile="$packageDir/$packageName.spec"
        local packageSummary="OpenMP Extras provides openmp and flang libraries."
        local packageSummaryLong="openmp-extras $packageVersion is based on LLVM 17 and is used for offloading to Radeon GPUs."
        local rpmRequires="rocm-llvm, rocm-device-libs, rocm-core"
        if [ "$packageName" == "openmp-extras-runtime" ]; then
          packageType="runtime"
          if [ "$STATIC_PKG_DEPS" == "OFF" ]; then
            rpmRequires="rocm-core, hsa-rocr"
          else
            rpmRequires="rocm-core, hsa-rocr-static-devel"
          fi
        else
          local rpmProvides="openmp-extras"
          local rpmObsoletes="openmp-extras"
          packageType="devel"
          if [ "$STATIC_PKG_DEPS" == "OFF" ]; then
            rpmRequires="$rpmRequires, openmp-extras-runtime, hsa-rocr-devel"
          else
            rpmRequires="$rpmRequires, openmp-extras-runtime, hsa-rocr-static-devel"
          fi
        fi

        if [ "$packageType" == "runtime" ]; then
          rm -rf "$packageDir"
          rm -rf "$RPM_PATH"
          mkdir -p "$RPM_PATH"
        fi
        echo RPM_PATH: $RPM_PATH
        echo mkdir -p $(dirname $specFile)
        mkdir -p "$(dirname $specFile)"

        {
          echo "%define is_runtime %( if [ $packageType == runtime ]; then echo "1" ; else echo "0"; fi )"
          echo "%define is_devel %( if [ $packageType == devel ]; then echo "1" ; else echo "0"; fi )"

          echo "Name:       $packageName"
          echo "Version:    $packageVersion"
          echo "Release:    ${CPACK_RPM_PACKAGE_RELEASE}%{?dist}"
          echo "Summary:    $packageSummary"
          echo "Group:      System Environment/Libraries"
          echo "License:    MIT and ASL 2.0 and ASL 2.0 with exceptions"
          echo "Vendor:     Advanced Micro Devices, Inc."
          echo "Prefix:     $INSTALL_PREFIX"
          echo "Requires:   $rpmRequires"
          echo "%if %is_devel"
          echo "Provides:   $rpmProvides"
          echo "Obsoletes:  $rpmObsoletes"
          echo "%endif"
          echo "%define debug_package %{nil}"
          echo "%define __os_install_post %{nil}"
          echo "%description"
          echo "$packageSummaryLong"

          echo "%prep"
          echo "%setup -T -D -c -n $packageName"
          echo "%build"

          echo "%install"
          echo "if [ -f $BUILD_PATH/build/installed_files.txt ] && [ ! -d $INSTALL_PREFIX/openmp-extras/devel ]; then"
          echo "  %if %is_runtime"
          echo "    mkdir -p \$RPM_BUILD_ROOT/openmp-extras"
          echo "  %else"
          echo "    mkdir -p \$RPM_BUILD_ROOT$copyPath/bin"
          echo "    mkdir -p \$RPM_BUILD_ROOT$installPath/lib/clang/$llvm_ver/include"
          echo "  %endif"
          echo "else"
          echo "  %if %is_runtime"
          echo "    mkdir -p  \$RPM_BUILD_ROOT$installPath"
          echo "  %else"
          echo "    rm -rf \$RPM_BUILD_ROOT/openmp-extras$installPath/*"
          echo "    echo mkdir -p \$RPM_BUILD_ROOT$copyPath/bin"
          echo "    mkdir -p \$RPM_BUILD_ROOT$copyPath/bin"
          echo "    mkdir -p \$RPM_BUILD_ROOT$installPath/lib/clang/$llvm_ver/include"
          echo "  %endif"
          echo "fi"

          echo "if [ -f $BUILD_PATH/build/installed_files.txt ] && [ ! -d $INSTALL_PREFIX/openmp-extras/devel ]; then"
          echo "  %if %is_runtime"
          echo "    cat $BUILD_PATH/build/installed_files.txt | grep -P '\.so|\.a' | cut -d':' -f2 | cut -d' ' -f2 | xargs -I {} cp -d --parents {} \$RPM_BUILD_ROOT"

          echo "    cp -d --parents "$installPath/lib/libgomp.so" \$RPM_BUILD_ROOT"
          echo "    cp -d --parents "$installPath/lib/libiomp5.so" \$RPM_BUILD_ROOT"
          echo "    cp -d --parents "$installPath/lib-debug/libgomp.so" \$RPM_BUILD_ROOT"
          echo "    cp -d --parents "$installPath/lib-debug/libiomp5.so" \$RPM_BUILD_ROOT"
          echo "  %endif"
          echo "%if %is_devel"
          echo "  cat "$BUILD_PATH"/build/installed_files.txt | grep -Pv '\.so|\.a' | cut -d':' -f2 | cut -d' ' -f2 | xargs -I {} cp -d --parents {} \$RPM_BUILD_ROOT"
          echo "%endif"

          echo "else"
          echo "  cp -r $AOMP/$packageType/* \$RPM_BUILD_ROOT$installPath"
          echo "  %if %is_devel"
          echo "    rm -rf \$RPM_BUILD_ROOT$installPath/share"
          echo "  %endif"
          echo "fi"

          echo "%if %is_devel"
          echo "  if [ ! -h \$RPM_BUILD_ROOT$copyPath/bin/aompcc ] && [ -e \$RPM_BUILD_ROOT$installPath/bin/aompcc ]; then"
          echo "    ln -s ../lib/llvm/bin/aompcc \$RPM_BUILD_ROOT$copyPath/bin/aompcc"
          echo "  fi"
          echo "  if [ -e \$RPM_BUILD_ROOT$installPath/bin/mymcpu ]; then"
          echo "    ln -s ../lib/llvm/bin/mymcpu \$RPM_BUILD_ROOT$copyPath/bin/mymcpu"
          echo "  fi"
          echo "  if [ -e \$RPM_BUILD_ROOT$installPath/bin/mygpu ]; then"
          echo "    ln -s ../lib/llvm/bin/mygpu \$RPM_BUILD_ROOT$copyPath/bin/mygpu"
          echo "  fi"
          echo "  ls \$RPM_BUILD_ROOT$copyPath"

          echo "  ln -s ../../../../include/omp.h  \$RPM_BUILD_ROOT/$installPath/lib/clang/$llvm_ver/include/omp.h"
          echo "  ln -s ../../../../include/ompt.h  \$RPM_BUILD_ROOT/$installPath/lib/clang/$llvm_ver/include/ompt.h"
          echo "  ln -s ../../../../include/omp-tools.h \$RPM_BUILD_ROOT/$installPath/lib/clang/$llvm_ver/include/omp-tools.h"
          echo "%endif"
          echo 'find $RPM_BUILD_ROOT \! -type d | sed "s|$RPM_BUILD_ROOT||"> files.list'

          echo "%if %is_runtime"
          echo "  mkdir -p \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras"
          echo "  cp -r $AOMP_REPOS/aomp/LICENSE \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras/LICENSE.apache2"
          echo "  cp -r $AOMP_REPOS/aomp-extras/LICENSE \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras/LICENSE.mit"
          echo "  cp -r $AOMP_REPOS/flang/LICENSE.txt \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras/LICENSE.flang"
          echo "%else"
          echo "  mkdir -p \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples"
          echo "  cp -r $AOMP_REPOS/aomp/examples/fortran \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples"
          echo "  cp -r $AOMP_REPOS/aomp/examples/openmp \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples"
          echo "  cp -r $AOMP_REPOS/aomp/examples/tools \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples"
          if [ -e "$AOMP_REPOS/aomp/examples/Makefile.help" ]; then
            echo "  cp $AOMP_REPOS/aomp/examples/Makefile* \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples"
          fi
          clean_examples \$RPM_BUILD_ROOT$copyPath/share/openmp-extras/examples
          echo "%endif"
          echo "%clean"
          echo "rm -rf \$RPM_BUILD_ROOT"

          echo "%files -f files.list"
          echo "%if %is_runtime"
          echo "  $copyPath/share/doc/openmp-extras"
          echo "%else"
          echo "  $copyPath/share/openmp-extras"
          echo "%endif"
          echo "%defattr(-,root,root,-)"
          echo "%if %is_runtime || %is_devel"
          echo "  $copyPath"
          echo "%endif"

        } > $specFile
        rpmbuild --define "_topdir $packageRpm" -ba $specFile
        mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH
}

package_openmp_extras_asan_rpm() {
        local packageName=$1
        local packageRpm="$packageDir/rpm"
        local specFile="$packageDir/$packageName.spec"
        local packageSummary="AddressSanitizer OpenMP Extras provides instrumented openmp and flang libraries."
        local packageSummaryLong="openmp-extras $packageVersion is based on LLVM 17 and is used for offloading to Radeon GPUs."
        local rpmRequires="hsa-rocr-asan, rocm-core-asan"
        local asanLibDir="runtime"

        rm -rf "$packageDir"
        rm -rf "$RPM_PATH"
        mkdir -p "$RPM_PATH"
        echo RPM_PATH: $RPM_PATH
        echo mkdir -p $(dirname $specFile)
        mkdir -p "$(dirname $specFile)"

        {
          echo "Name:       $packageName"
          echo "Version:    $packageVersion"
          echo "Release:    ${CPACK_RPM_PACKAGE_RELEASE}%{?dist}"
          echo "Summary:    $packageSummary"
          echo "Group:      System Environment/Libraries"
          echo "License:    MIT and ASL 2.0 and ASL 2.0 with exceptions"
          echo "Vendor:     Advanced Micro Devices, Inc."
          echo "Requires:   $rpmRequires"
          echo "%define __os_install_post %{nil}"
          echo "%description"
          echo "%undefine _debugsource_packages"
          echo "$packageSummaryLong"

          echo "%prep"
          echo "%setup -T -D -c -n $packageName"
          echo "%build"

          echo "%install"
          echo "if [ -f $BUILD_PATH/build/installed_files.txt ] && [ ! -d "$INSTALL_PREFIX"/openmp-extras ]; then"
          echo "  cat $BUILD_PATH/build/installed_files.txt | grep -P 'asan' | cut -d':' -f2 | cut -d' ' -f2 | xargs -I {} cp -d --parents {} \$RPM_BUILD_ROOT"
          echo "  cp -d --parents "$installPath/lib/asan/libgomp.so" \$RPM_BUILD_ROOT"
          echo "  cp -d --parents "$installPath/lib/asan/libiomp5.so" \$RPM_BUILD_ROOT"
          echo "  cp -d --parents "$installPath/lib-debug/asan/libgomp.so" \$RPM_BUILD_ROOT"
          echo "  cp -d --parents "$installPath/lib-debug/asan/libiomp5.so" \$RPM_BUILD_ROOT"
          echo "else"
          echo "  mkdir -p  \$RPM_BUILD_ROOT$installPath/lib/asan"
          echo "  mkdir -p  \$RPM_BUILD_ROOT$installPath/lib-debug/asan"
          echo "  cp -r $AOMP/lib/asan/* \$RPM_BUILD_ROOT$installPath/lib/asan"
          echo "  cp -r $AOMP/lib-debug/asan/* \$RPM_BUILD_ROOT$installPath/lib-debug/asan"
          echo "  cp -r $AOMP/$asanLibDir/lib/asan/* \$RPM_BUILD_ROOT$installPath/lib/asan"
          echo "  cp -r $AOMP/$asanLibDir/lib-debug/asan/* \$RPM_BUILD_ROOT$installPath/lib-debug/asan"
          echo "  cp -r $AOMP/devel/lib/asan/* \$RPM_BUILD_ROOT$installPath/lib/asan"
          echo "  cp -r $AOMP/devel/lib-debug/asan/* \$RPM_BUILD_ROOT$installPath/lib-debug/asan"
          echo "fi"

          echo 'find $RPM_BUILD_ROOT \! -type d | sed "s|$RPM_BUILD_ROOT||"> files.list'

          echo "  mkdir -p \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras-asan"
          echo "  cp -r $AOMP_REPOS/aomp/LICENSE \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras-asan/LICENSE.apache2"
          echo "  cp -r $AOMP_REPOS/aomp-extras/LICENSE \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras-asan/LICENSE.mit"
          echo "  cp -r $AOMP_REPOS/flang/LICENSE.txt \$RPM_BUILD_ROOT$copyPath/share/doc/openmp-extras-asan/LICENSE.flang"
          echo "%clean"
          echo "rm -rf \$RPM_BUILD_ROOT"

          echo "%files -f files.list"
          echo "  $copyPath/share/doc/openmp-extras-asan"
          echo "%defattr(-,root,root,-)"
          echo "  $copyPath"

        } > $specFile
        rpmbuild --define "_topdir $packageRpm" -ba $specFile
        mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH
}

package_openmp_extras() {
    local DISTRO_NAME=$(cat /etc/os-release | grep -e ^NAME=)
    local installPath="$ROCM_INSTALL_PATH/lib/llvm"
    local copyPath="$ROCM_INSTALL_PATH"
    local packageDir="$BUILD_PATH/package"
    local llvm_ver=`$INSTALL_PREFIX/lib/llvm/bin/clang --print-resource-dir | sed 's^/llvm/lib/clang/^ ^' | awk '{print $2}'`
    local debNames="openmp-extras-runtime openmp-extras-dev"
    local rpmNames="openmp-extras-runtime openmp-extras-devel"
    if [ "$SANITIZER" == "1" ]; then
      local asanPkgName="openmp-extras-asan"
      if [[ $DISTRO_NAME =~ "Ubuntu" ]]; then
        package_openmp_extras_asan_deb $asanPkgName
      else
        package_openmp_extras_asan_rpm $asanPkgName
      fi
      return 0
    fi

    if [[ $DISTRO_NAME =~ "Ubuntu" ]]; then
      for name in $debNames; do
        package_openmp_extras_deb $name
      done
    else
      for name in $rpmNames; do
        package_openmp_extras_rpm $name
      done
    fi
}

package_tests_deb(){
    local packageDir="$BUILD_PATH/package"
    local packageDeb="$packageDir/deb"
    local packageArch="amd64"
    local packageMaintainer="Openmp Extras Support <openmp-extras.support@amd.com>"
    local packageSummary="Tests for openmp-extras."
    local packageSummaryLong="Tests for openmp-extras $packageMajorVersion-$packageMinorVersion is based on LLVM 17 and is used for offloading to Radeon GPUs."
    local debDependencies="openmp-extras-dev, rocm-core"
    local debRecommends=""
    local controlFile="$packageDeb/openmp-extras/DEBIAN/control"
    local installPath="$ROCM_INSTALL_PATH/share/openmp-extras/tests"
    local packageName="openmp-extras-tests"

    rm -rf "$packageDir"

    mkdir -p $packageDeb/openmp-extras"$installPath"
    if [ -e $(dirname $controlFile) ]; then
        rm $(dirname $controlFile)
    fi
    mkdir -p "$(dirname $controlFile)"
    cp -r "$AOMP_REPOS/aomp/." "$packageDeb/openmp-extras/$installPath"
    rm -rf "$packageDeb"/openmp-extras"$installPath"/.git "$packageDeb"/openmp-extras"$installPath"/.github
    cp "$OUT_DIR/build/lightning/bin/FileCheck" "$packageDeb/openmp-extras/$installPath/bin"
    {
      echo "Package: $packageName"
      echo "Architecture: $packageArch"
      echo "Section: devel"
      echo "Priority: optional"
      echo "Maintainer: $packageMaintainer"
      echo "Version: $packageVersion-${CPACK_DEBIAN_PACKAGE_RELEASE}"
      echo "Depends: $debDependencies"
      echo "Recommends: $debRecommends"
      echo "Description: $packageSummary"
      echo "  $packageSummaryLong"
    } > $controlFile
    fakeroot dpkg-deb -Zgzip --build $packageDeb/openmp-extras \
    "${DEB_PATH}/${packageName}_${packageVersion}-${CPACK_DEBIAN_PACKAGE_RELEASE}_${packageArch}.deb"
}

package_tests_rpm(){
    AOMP_STANDALONE_BUILD=1 $AOMP_REPOS/aomp/bin/build_fixups.sh
    local copyPath="$ROCM_INSTALL_PATH"
    local packageDir="$BUILD_PATH/package"
    local packageRpm="$packageDir/rpm"
    local installPath="$ROCM_INSTALL_PATH/share/openmp-extras/tests"
    local packageName="openmp-extras-tests"
    local specFile="$packageDir/$packageName.spec"
    local packageSummary="Tests for openmp-extras."
    local packageSummaryLong="Tests for openmp-extras $packageVersion is based on LLVM 18 and is used for offloading to Radeon GPUs."

    rm -rf "$packageDir"
    mkdir -p "$packageRpm/openmp-extras/$installPath"
    {
      echo "AutoReqProv: no"
      echo "Name:       $packageName"
      echo "Version:    $packageVersion"
      echo "Release:    ${CPACK_RPM_PACKAGE_RELEASE}%{?dist}"
      echo "Summary:    $packageSummary"
      echo "Group:      System Environment/Libraries"
      echo "License:    Advanced Micro Devices, Inc."
      echo "Vendor:     Advanced Micro Devices, Inc."
      echo "Prefix:     $INSTALL_PREFIX"
      echo "Requires:   $rpmRequires"
      echo "%define debug_package %{nil}"
      # Redefining __os_install_post to remove stripping
      echo "%define __os_install_post %{nil}"
      echo "%description"
      echo "$packageSummaryLong"

      echo "%prep"
      echo "%setup -T -D -c -n $packageName"
      echo "%build"

      echo "%install"
      echo "mkdir -p  \$RPM_BUILD_ROOT$installPath"
      echo "cp -R $AOMP_REPOS/aomp/. \$RPM_BUILD_ROOT$installPath"
      echo "rm -rf \$RPM_BUILD_ROOT$installPath/.git \$RPM_BUILD_ROOT$installPath/.github"
      echo "cp $OUT_DIR/build/lightning/bin/FileCheck \$RPM_BUILD_ROOT$installPath/bin"
      echo 'find $RPM_BUILD_ROOT \! -type d | sed "s|$RPM_BUILD_ROOT||"> files.list'

      echo "%clean"
      echo "rm -rf \$RPM_BUILD_ROOT"

      echo "%files -f files.list"
      echo "$installPath"
      echo "%defattr(-,root,root,-)"

      echo "%postun"
      echo "rm -rf $installPath"
    } > $specFile
    rpmbuild --define "_topdir $packageRpm" -ba $specFile
    mv $packageRpm/RPMS/x86_64/*.rpm $RPM_PATH
}

package_tests() {
    local DISTRO_NAME=$(cat /etc/os-release | grep -e ^NAME=)
    if [[ $DISTRO_NAME =~ "Ubuntu" ]]; then
        package_tests_deb
    else
        package_tests_rpm
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

case $TARGET in
    (clean) clean_openmp_extras ;;
    (build) build_openmp_extras; package_openmp_extras; package_tests ;;
    (outdir) print_output_directory ;;
    (*) die "Invalid target $TARGET" ;;
esac

echo "Operation complete"
