#!/bin/bash

: <<'END_COMMENT'
    This script retains only those functions from the original compute_utils.sh
    that the ROCm build scripts utilize.
END_COMMENT

set -e
set -o pipefail


# Set a sensible default value for DASH_JAY in case none is provided
DASH_JAY=${DASH_JAY:-"-j $(nproc)"}

# Enable ccache by default unless requested otherwhise
if [[ "$ROCM_USE_CCACHE" != "0" ]] ; then
	for d in /usr/lib/ccache /usr/lib64/ccache ;do
		if [ -d "$d" ]; then
			PATH="$d:$PATH"
			break # Only add one ccache at most
		fi
	done
fi

# Print message to stderr
#   param message string to print on exit
# Example: printErr "file not found"
printErr() {
    echo "$@" 1>&2
}

# Print message to stderr and terminate current program
#   param message string to print on exit
# Example: die "Your program" has terminated
die() {
    printErr "FATAL: $@"
    exit 1
}

# Die if first argument is empty
#   param string to validate
#   param error message
# Example: die "$VARIABLE" "Your program" has terminated
dieIfEmpty() {
    if [ "$1" == "" ]; then
        shift
        die "$@"
    fi
}

# Get directory with build output package
# Precedence:
#       1. PWD
#       2. Caller's folder
#       3. Known build output folder
getPackageRoot() {
    local scriptPath=$(readlink -f $(dirname $BASH_SOURCE))
    local testFile="build.version"

    if [ -a "$PWD/$testFile" ]; then
        echo "$PWD"
    elif [ -a "$scriptPath/../$testFile" ]; then
        echo "$scriptPath/.."
    elif [ -a "$scriptPath/$testFile" ]; then
        echo "$scriptPath"
    elif [ ! -z "$OUT_DIR" ]; then
        echo "$OUT_DIR"
    else
        die "Failed to determine package directory"
    fi
}

# Get a list of directories containing the build output
# shared objects.
# Important: PWD takes precedence over build output folder
getLibPath() {
    local packageRoot="$(getPackageRoot)"
    dieIfEmpty "$packageRoot"
    echo "$packageRoot/lib"
}

# Get a list of directories containing the output executables
#     param binDir (optional) - package name
# Important: PWD takes precedence over build output folder
getBinPath() {
    local binDir="$1"
    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"

    if [ "$binDir" == "" ]; then
        echo "$packageRoot/bin"
    else
        echo "$packageRoot/bin/$binDir"
    fi
}

# Get a list of directories containing the output source files
# Important: PWD takes precedence over build output folder
getSrcPath() {
    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"
    echo "$packageRoot/src"
}

# Get a list of directories to place build output
#   param moduleName - name of the module for the build path
# Important: PWD takes precedence over build output folder
getBuildPath() {
    local moduleName="$1"
    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"
    echo "$packageRoot/build/$moduleName"
}

# Get a list of directories containing the output etc files
# Important: PWD takes precedence over build output folder
getUtilsPath() {
    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"
    echo "$packageRoot/utils"
}

# Get a list of directories containing the output include files
# Important: PWD takes precedence over build output folder
getIncludePath() {
    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"
    echo "$packageRoot/include"
}

# Get the directory containing the cmake config files
getCmakePath() {
    local rocmInstallPath=${ROCM_INSTALL_PATH}
    local cmakePath="lib/cmake"
    if [ "$ASAN_CMAKE_PARAMS" == "true" ] ; then
        cmakePath="lib/asan/cmake"
    fi
    dieIfEmpty "$rocmInstallPath"
    echo "$rocmInstallPath/$cmakePath"
}

# Get a list of directories containing the output debian files
# Important: PWD takes precedence over build output folder
getDebPath() {
    local packageName="$1"
    dieIfEmpty "$packageName" "No valid package name specified"

    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"

    echo "$packageRoot/deb/$packageName"
}

# Get a list of directories containing the output rpm files
# Important: PWD takes precedence over build output folder
getRpmPath() {
    local packageName="$1"
    dieIfEmpty "$packageName" "No valid package name specified"

    local packageRoot=$(getPackageRoot)
    dieIfEmpty "$packageRoot"

    echo "$packageRoot/rpm/$packageName"
}



verifyEnvSetup() {
    if [ -z "$OUT_DIR" ]; then
        die "Please source build/envsetup.sh first."
    fi
}

# Copy a file or directory to target location and show single line progress
progressCopy() {
    if [ -d "$1" ]; then
        rsync -a "$1"/* "$2"
    else
        rsync -a "$1" "$2"
    fi
}


#following three common functions have been written to addition of static libraries
print_lib_type() {
   if [ "$1" == "OFF" ];
   then
       echo " Building Archive "
   else
       echo " Building Shared Object "
   fi
}

check_conflicting_options() {
    # 1->CLEAN_OR_OUT 2->PKGTYPE 3->MAKETARGET
    RET_CONFLICT=0
    if [ $1 -ge 2 ]; then
       if [ "$2" != "deb" ] && [ "$2" != "rpm" ] && [ "$2" != "tar" ]; then
          echo " Wrong Param Passed for Package Type for the Outdir... "
          RET_CONFLICT=30
       fi
    fi
    # check Clean Vs Outdir
    if [ $1 -ge 3 ] && [ $RET_CONFLICT -eq 0 ] ; then
       echo " Clean & Out Both are sepcified.  Not accepted. Bailing .. "
       RET_CONFLICT=40
    fi
    if [ $RET_CONFLICT -eq 0 ] && [ "$3" != "deb" ] && [ "$3" != "rpm" ] && [ "$3" != "all" ] && [ "$3" != "tar" ]; then
       echo " Wrong Param Passed for Package Type... "
       RET_CONFLICT=50
    fi
}

set_asan_env_vars() {
    # Flag to set cmake build params for ASAN builds
    ASAN_CMAKE_PARAMS="true"
    # Pass the LLVM bin path as the first parameter
    LLVM_BIN_DIR=${1:-"${ROCM_INSTALL_PATH}/llvm/bin"}
    export CC="$LLVM_BIN_DIR/clang"
    export CXX="$LLVM_BIN_DIR/clang++"
    export PATH="$PATH:$LLVM_BIN_DIR/"
    # get exact path to ASAN lib containing clang version
    ASAN_LIB_PATH=$(clang --print-file-name=libclang_rt.asan-x86_64.so)
    export LD_LIBRARY_PATH="${ASAN_LIB_PATH%/*}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ASAN_OPTIONS="detect_leaks=0"
}

set_address_sanitizer_on() {
    # In SLES and RHEL debuginfo package is not getting generated
    # when compiler is set to clang. Using default -gdwarf-5 is getting unhandled in these distros
    # So setting -gdwarf-4 as a quick fix
    # TODO: -gdwarf-5 unhandling when compiler set to clang need to be fixed
    SET_DWARF_VERSION_4=""
    if [[ $DISTRO_ID == sles* ]] || [[ $DISTRO_ID == rhel* ]]; then
        SET_DWARF_VERSION_4="-gdwarf-4"
    fi
    export CFLAGS="-fsanitize=address -shared-libasan -g $SET_DWARF_VERSION_4"
    export CXXFLAGS="-fsanitize=address -shared-libasan -g $SET_DWARF_VERSION_4"
    export LDFLAGS="-Wl,--enable-new-dtags -fuse-ld=lld  -fsanitize=address -shared-libasan -g -Wl,--build-id=sha1 -L$ROCM_INSTALL_PATH/lib/asan"
}

ack_and_ignore_asan() {
    echo "-a parameter accepted but ignored"
}

ack_and_skip_static() {
    echo "-s parameter accepted but static build is not enabled for this component..skipping"
    exit 0
}

#debug function #dumping values in case of error to solve the same
print_vars() {
echo " Status of Vars in $1 build "
echo " TARGET= $2 "
echo " BUILD_TYPE = $3 "
echo " SHARED_LIBS = $4 "
echo " CLEAN_OR_OUT = $5 "
echo " PKGTYPE= $6 "
echo " MAKETARGET = $7 "
}

# Provide this as a function, rather than a variable to delay the evaluation
# of variables. In particular we might want to put code in here which changes
# depending on if we are building with the address sanitizer or not
# Can do things like set the packaging type - no point in packaging RPM on debian and
# vica versa.
# Set CPACK_RPM_INSTALL_WITH_EXEC so it packages debug info for shared libraries.
rocm_common_cmake_params(){
    if [ "$BUILD_TYPE" = "RelWithDebInfo" ] ; then
	printf '%s ' "-DCPACK_RPM_DEBUGINFO_PACKAGE=TRUE" \
	       "-DCPACK_DEBIAN_DEBUGINFO_PACKAGE=TRUE" \
	       "-DCPACK_RPM_INSTALL_WITH_EXEC=TRUE" \
               # end of list comment or blank line
    fi
    printf '%s ' "-DROCM_DEP_ROCMCORE=ON" \
                 "-DCMAKE_EXE_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_EXE_RPATH" \
                 "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_LIB_RPATH" \
                 "-DFILE_REORG_BACKWARD_COMPATIBILITY=OFF" \
                 "-DCPACK_RPM_PACKAGE_RELOCATABLE=ON" \
                 "-DCPACK_SET_DESTDIR=OFF" \
                 "-DINCLUDE_PATH_COMPATIBILITY=OFF" \
    # set lib directory to lib/asan for ASAN builds
    # Disable file reorg backward compatibilty for ASAN builds
    # ENABLE_ASAN_PACKAGING - Used for enabling ASAN packaging
    if [ "$ASAN_CMAKE_PARAMS" == "true" ] ; then
        local asan_common_cmake_params
        local ASAN_LIBDIR="lib/asan"
        local CMAKE_PATH=$(getCmakePath)
        asan_common_cmake_params=(
            "-DCMAKE_INSTALL_LIBDIR=$ASAN_LIBDIR"
            "-DCMAKE_PREFIX_PATH=$CMAKE_PATH;${ROCM_INSTALL_PATH}/$ASAN_LIBDIR;$ROCM_INSTALL_PATH/llvm;$ROCM_INSTALL_PATH"
            "-DENABLE_ASAN_PACKAGING=$ASAN_CMAKE_PARAMS"
            "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ROCM_ASAN_LIB_RPATH"
        )
        printf '%s ' "${asan_common_cmake_params[@]}"
    else
        printf '%s ' "-DCMAKE_INSTALL_LIBDIR=lib" \
        # end of list comment or blank line
    fi
}

rocm_cmake_params() {
    local cmake_params

    cmake_params=(
        "-DCMAKE_PREFIX_PATH=${ROCM_INSTALL_PATH}/llvm;${ROCM_INSTALL_PATH}"
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-'RelWithDebInfo'}"
        "-DCMAKE_VERBOSE_MAKEFILE=1"
        "-DCPACK_GENERATOR=${CPACKGEN:-'DEB;RPM'}"
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"
        "-DROCM_PATCH_VERSION=${ROCM_LIBPATCH_VERSION}"
        "-DCMAKE_INSTALL_PREFIX=${ROCM_INSTALL_PATH}"
        "-DCPACK_PACKAGING_INSTALL_PREFIX=${ROCM_INSTALL_PATH}"
    )

    printf '%s ' "${cmake_params[@]}"
}

copy_if(){
    local type=$1 selector=$2 dir=$3
    shift 3
    mkdir -p "$dir"
    if [[ "$selector" =~ "$type" ]] ; then
	cp -a "$@" "$dir"
    fi
    # handle ddeb files as well, renaming them on the way
    for f
    do
	case "$f" in
	    # Properly formed debian package name is a number of _ separated fields
	    # The first is the package name.
	    # Second is version number
	    # third is architecture
	    # Ensure we have at least one _ in the name
	    (*"_"*".deb")
		local deb=${f%.deb}
		local basename=${deb##*/}
		local dirname=${f%/*}
                # filename($f) can be either /some/path/pkgname.deb or pkgname.deb
                # If its pkgname.deb, then directory should be .
                [[ "$dirname" == "$f" ]] && dirname=.
		local pkgname=${basename%%_*}
		local pkgextra=${basename#*_}
		# cmake 3.22 creates the filename by replacing .deb with -dbgsym.ddeb
		# at least for hostcall. Mind you hostcall looks to be incorrectly packaged.
		if [ -e "${deb}-dbgsym.ddeb" ]
		then
		    dest=${deb##*/}
		    dest="${dest%%_*}-dbgsym_${dest#*_}"
		    cp -a "${deb}-dbgsym.ddeb" "$dir/${dest##*/}.deb"
		fi
		# This is needed for comgr
		if [ -e "$dirname/${pkgname}-dbgsym_${pkgextra}.ddeb" ]
		then
		    cp "$dirname/${pkgname}-dbgsym_${pkgextra}.ddeb" "$dir/${pkgname}-dbgsym_${pkgextra}.deb"
		fi
		;;
	esac
    done
}


# Function to remove -r or -R from MAKEFLAGS
remove_make_r_flags(){
    local firstword='^[^ ]*'
    if [[ "$MAKEFLAGS" =~ ${firstword}r ]] ; then MAKEFLAGS=${MAKEFLAGS/r/} ; fi
    if [[ "$MAKEFLAGS" =~ ${firstword}R ]] ; then MAKEFLAGS=${MAKEFLAGS/R/} ; fi
}

# Set a variable to the value needed by cmake to use ninja if it is available
# If GEN_NINJA is already defined, even as the empty string, then leave the value alone
# Intended use in build_xxxx.sh is ${GEN_NINJA:+"$GEN_NINJA"} to cope with potentially weird values
# but in practice just ${GEN_NINJA} without quotes will be fine.
# e.g.            cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $GEN_NINJA
# If for some reason you wanted to build without ninja just export an empty GEN_NINJA variable
if [ "${GEN_NINJA+defined}" != "defined" ] && command -v ninja >/dev/null ; then
    GEN_NINJA=-GNinja
fi

