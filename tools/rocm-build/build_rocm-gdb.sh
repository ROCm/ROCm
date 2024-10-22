#!/bin/bash

source "${BASH_SOURCE%/*}/compute_utils.sh" || return

remove_make_r_flags

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
    echo "  -h,  --help               Prints this help"
    echo "  -s,  --static             Supports static CI by accepting this param & not bailing out. No effect of the param though"
    echo
    echo "Possible values for <type>:"
    echo "  deb -> Debian format (default)"
    echo "  rpm -> RPM format"
    echo

    return 0
}

toStdoutStderr(){
    printf '%s\n' "$@" >&2
    printf '%s\n' "$@"
}

linkFiles(){
    cp -lfR "$1" "$2" || cp -fR "$1" "$2"
}

PROJ_NAME=rocm-gdb
TARGET=build
MAKETARGET=deb
BUILD_DIR=$(getBuildPath $PROJ_NAME)
PACKAGE_DEB=$(getPackageRoot)/deb/$PROJ_NAME
PACKAGE_RPM=$(getPackageRoot)/rpm/$PROJ_NAME
MAKE_OPTS="$DASH_JAY"
BUG_URL="https://github.com/ROCm-Developer-Tools/ROCgdb/issues"
SHARED_LIBS="ON"
CLEAN_OR_OUT=0;
MAKETARGET="deb"
PKGTYPE="deb"
LDFLAGS="$LDFLAGS -Wl,--enable-new-dtags"


tokeep=(
    main${ROCM_INSTALL_PATH}/bin/rocgdb
    main${ROCM_INSTALL_PATH}/bin/roccoremerge
    main${ROCM_INSTALL_PATH}/share/rocgdb/python/gdb/.*
    main${ROCM_INSTALL_PATH}/share/rocgdb/syscalls/amd64-linux.xml
    main${ROCM_INSTALL_PATH}/share/rocgdb/syscalls/gdb-syscalls.dtd
    main${ROCM_INSTALL_PATH}/share/rocgdb/syscalls/i386-linux.xml
    main${ROCM_INSTALL_PATH}/share/doc/rocgdb/NOTICES.txt
    main${ROCM_INSTALL_PATH}/share/doc/rocgdb/rocannotate.pdf
    main${ROCM_INSTALL_PATH}/share/doc/rocgdb/rocgdb.pdf
    main${ROCM_INSTALL_PATH}/share/doc/rocgdb/rocrefcard.pdf
    main${ROCM_INSTALL_PATH}/share/doc/rocgdb/rocstabs.pdf
    main${ROCM_INSTALL_PATH}/share/info/rocgdb/dir
    main${ROCM_INSTALL_PATH}/share/info/rocgdb/annotate.info
    main${ROCM_INSTALL_PATH}/share/info/rocgdb/gdb.info
    main${ROCM_INSTALL_PATH}/share/info/rocgdb/stabs.info
    main${ROCM_INSTALL_PATH}/share/man/man1/rocgdb.1
    main${ROCM_INSTALL_PATH}/share/man/man1/roccoremerge.1
    main${ROCM_INSTALL_PATH}/share/man/man5/rocgdbinit.5
    main${ROCM_INSTALL_PATH}/share/html/rocannotate/.*
    main${ROCM_INSTALL_PATH}/share/html/rocgdb/.*
    main${ROCM_INSTALL_PATH}/share/html/rocstabs/.*
)

keep_wanted_files(){
    (
        cd "$BUILD_DIR/package/"
        printf -v keeppattern '%s\n' "${tokeep[@]}"
        find main/opt -type f | grep -xv "$keeppattern" | xargs -r rm
        find main/opt -type d -empty -delete
    )
    return 0
}

copy_testsuite_files() {
(
dest="$BUILD_DIR/package/tests${ROCM_INSTALL_PATH}/test/gdb/"
cd "$ROCM_GDB_ROOT"
find \
    config.guess \
    config.sub \
    contrib/dg-extract-results.py \
    contrib/dg-extract-results.sh \
    gdb/contrib \
    gdb/disable-implicit-rules.mk \
    gdb/features \
    gdb/silent-rules.mk \
    gdb/testsuite \
    include/dwarf2.def \
    include/dwarf2.h \
    install-sh \
    -print0 | cpio -pdu0 "$dest"
)
}

clean() {
    echo "Cleaning $PROJ_NAME"

    rm -rf $BUILD_DIR
    rm -rf $PACKAGE_DEB
    rm -rf $PACKAGE_RPM
}

get_version(){
    VERSION=$(sed -n 's/^.*char version[^"]*"\([^"]*\)".*;.*/\1/p' $BUILD_DIR/gdb/version.c || : )
    VERSION=${VERSION:-$1}
}

package_deb(){
    mkdir -p "$BUILD_DIR/package/main/DEBIAN"
    local VERSION
    get_version unknown
    VERSION="${VERSION}.${ROCM_LIBPATCH_VERSION}"

    grep -v '^# ' > "$BUILD_DIR/package/main/DEBIAN/control" <<EOF
# Required fields
Version: ${VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}
Package: ${PROJ_NAME}
Maintainer: ROCm Debugger Support <rocm-gdb.support@amd.com>
Description: ROCgdb
 This is ROCgdb, the AMD ROCm source-level debugger for Linux,
 based on GDB, the GNU source-level debugger.
# Optional fields
Section: utils
Architecture: amd64
Essential: no
Priority: optional
Depends: libexpat1, libtinfo5, libncurses5, rocm-dbgapi, libpython3-all-dev, libbabeltrace-ctf1 (>= 1.2.1), libbabeltrace1 (>= 1.2.1), rocm-core
EOF

    mkdir -p "$OUT_DIR/deb/$PROJ_NAME"
    fakeroot dpkg-deb -Zgzip --build "$BUILD_DIR/package/main" "$OUT_DIR/deb/$PROJ_NAME"

    mkdir -p "$BUILD_DIR/package/tests/DEBIAN"
    mkdir -p "$BUILD_DIR/package/tests/${ROCM_INSTALL_PATH}/test/gdb"

    grep -v '^# ' > "$BUILD_DIR/package/tests/DEBIAN/control" <<EOF
# Required fields
Version: ${VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}
Package: ${PROJ_NAME}-tests
Maintainer: ROCm Debugger Support <rocm-gdb.support@amd.com>
Description: ROCgdb tests
 Test Suite for ROCgdb
# Optional fields
Section: utils
Architecture: amd64
Essential: no
Priority: optional
# rocm-core as policy says everything to depend on rocm-core
Depends: ${PROJ_NAME} (=${VERSION}-${CPACK_DEBIAN_PACKAGE_RELEASE}), dejagnu, rocm-core, make
EOF

    copy_testsuite_files
    fakeroot dpkg-deb -Zgzip --build "$BUILD_DIR/package/tests" "$OUT_DIR/deb/$PROJ_NAME"
}

package_rpm(){
    set -- rocm-gdb
    local packageDir="$BUILD_DIR/package_rpm/$1"
    local specFile="$packageDir/$1.spec"
    local packageRpm="$packageDir/rpm"

    local VERSION
    get_version 0.0.0

    VERSION=${VERSION}.${ROCM_LIBPATCH_VERSION}

    local ospost="$(echo '%define __os_install_post \'
                    rpm --showrc | sed '1,/^-14: __os_install_post/d;
                                       /^-14:/,$d;/^%{nil}/!s/$/ \\/;
                                       /brp-python-bytecompile/d')"

    echo "specFile:        $specFile"
    echo "packageRpm:      $packageRpm"

    mkdir -p "$packageDir"

    grep -v '^## ' <<- EOF > $specFile
## Set up where this stuff goes
%define _topdir $packageRpm
%define _rpmfilename %%{ARCH}/%%{NAME}-${VERSION}-${CPACK_RPM_PACKAGE_RELEASE}%{?dist}.%%{ARCH}.rpm
## The __os_install_post macro on centos creates .pyc and .pyo objects
## by calling brp-python-bytecompile
## This then creates an issue as the script doesn't package these files
## override it
$ospost
##
Name: ${PROJ_NAME}
Group: Development/Tools/Debuggers
Summary: ROCm source-level debugger for Linux
## rpm requires the version to be dot separated numbers
Version: ${VERSION//-/_}
Release: ${CPACK_RPM_PACKAGE_RELEASE}%{?dist}
License: GPL
Requires: rocm-core

%description
This is ROCgdb, the ROCm source-level debugger for Linux, based on
GDB, the GNU source-level debugger.

The ROCgdb documentation is available at:
https://github.com/RadeonOpenCompute/ROCm

## these things are commented out as they are not needed, but are
## left in for documentation.
# %prep
# : Should not need to do anything in prep
# %build
# : Should not need to do anything in build as make does that
# %clean
# : Should not need to do anything in clean
## This is the meat. Get a copy of the files from where we built them
## into the local RPM_BUILD_ROOT and left the defaults take over. Need
## to quote the dollar signs as we want rpm to expand them when it is
## run, rather than the shell when we build the spec file.
%install
rm -rf \$RPM_BUILD_ROOT
mkdir -p \$RPM_BUILD_ROOT
# Get a copy of the built tree.
cp -ar $BUILD_DIR/package/main/opt \$RPM_BUILD_ROOT/opt
## The file section is generated by walking the tree.
%files
EOF

    find $BUILD_DIR/package/main/opt -type d | sed "s:$BUILD_DIR/package/main:%dir :" >> $specFile
    find $BUILD_DIR/package/main/opt ! -type d | sed "s:$BUILD_DIR/package/main::" >> $specFile

    rpmbuild --define "_topdir $packageRpm" -ba $specFile

    mkdir -p "$PACKAGE_RPM"        # e.g. out/ubuntu-16.04/16.04/rpm/rocm-gdb
    mv $packageRpm/RPMS/x86_64/*.rpm "$PACKAGE_RPM"
}

package_rpm_tests(){
    set -- rocm-gdb-tests
    local packageDir="$BUILD_DIR/package_rpm/$1"
    local specFile="$packageDir/$1.spec"
    local packageRpm="$packageDir/rpm"

    local VERSION
    get_version 0.0.0
    VERSION=${VERSION}.${ROCM_LIBPATCH_VERSION}
    local RELEASE=${CPACK_RPM_PACKAGE_RELEASE}%{?dist}

    echo "specFile:        $specFile"
    echo "packageRpm:      $packageRpm"

    mkdir -p "$packageRpm"

    local ospost="$(echo '%define __os_install_post \'
                    rpm --showrc | sed '1,/^-14: __os_install_post/d;
                                       /^-14:/,$d;/^%{nil}/!s/$/ \\/;
                                       /brp-python-bytecompile/d')"
    grep -v '^## ' <<- EOF > $specFile
## Set up where this stuff goes
%define _topdir $packageRpm
%define _rpmfilename %%{ARCH}/%%{NAME}-${VERSION}-${RELEASE}.%%{ARCH}.rpm
## The __os_install_post macro on centos creates .pyc and .pyo objects
## by calling brp-python-bytecompile
## This then creates an issue as the script doesn't package these files
## override it
$ospost
##
Name: ${PROJ_NAME}-tests
Group: Development/Tools/Debuggers
Summary: Tests for gdb enhanced to debug AMD GPUs
Version: ${VERSION//-/_}
Release: ${RELEASE}
License: GPL
Requires: dejagnu, ${PROJ_NAME} = ${VERSION//-/_}-${RELEASE}, rocm-core, make

%description
Tests for ROCgdb

## these things are commented out as they are not needed, but are
## left in for documentation.
# %prep
# : Should not need to do anything in prep
# %build
# : Should not need to do anything in build as make does that
# %clean
# : Should not need to do anything in clean
## This is the meat. Get a copy of the files from where we built them
## into the local RPM_BUILD_ROOT and left the defaults take over. Need
## to quote the dollar signs as we want rpm to expand them when it is
## run, rather than the shell when we build the spec file.
%install
rm -rf \$RPM_BUILD_ROOT
mkdir -p \$RPM_BUILD_ROOT
# Get a copy of the built tree.
cp -ar $BUILD_DIR/package/tests/opt \$RPM_BUILD_ROOT/opt
## The file section is generated by walking the tree.
%files
## package everything in \$RPM_BUILD_ROOT/${ROCM_INSTALL_PATH}/test.
## A little excessive but this is just an internal test package
${ROCM_INSTALL_PATH}/test
EOF

    copy_testsuite_files

    find $BUILD_DIR/package/tests/opt -type f -exec sed -i '1s:^#! */usr/bin/python\>:&3:' {} +

    rpmbuild --define "_topdir $packageRpm" -ba $specFile
    mkdir -p "$PACKAGE_RPM"
    mv $packageRpm/RPMS/x86_64/*.rpm "$PACKAGE_RPM"
}

build() {
    if [ ! -e "$ROCM_GDB_ROOT/configure" ]
    then
        toStdoutStderr "No $ROCM_GDB_ROOT/configure file, skippping rocm-gdb"
        exit 0
    fi
    local pythonver=python3
    if [[ "$DISTRO_ID" == "ubuntu-18.04" ]]; then
        pythonver=python3.8
    fi

    if [[ "$DISTRO_ID" == "centos-9" ]] || [[ "$DISTRO_ID" == "rhel-9.0" ]]; then
        fmtutil-user --missing
    fi
    echo "Building $PROJ_NAME"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || die "Failed to cd to '$BUILD_DIR'"

    $ROCM_GDB_ROOT/configure --program-prefix=roc --prefix="${ROCM_INSTALL_PATH}" \
	--htmldir="\${prefix}/share/html" --pdfdir="\${prefix}/share/doc/rocgdb" \
	--infodir="\${prefix}/share/info/rocgdb" \
	--with-separate-debug-dir="\${prefix}/lib/debug:/usr/lib/debug" \
	--with-gdb-datadir="\${prefix}/share/rocgdb" --enable-64-bit-bfd \
        --with-bugurl="$BUG_URL" --with-pkgversion="${ROCM_BUILD_ID:-ROCm}" \
        --enable-targets="x86_64-linux-gnu,amdgcn-amd-amdhsa" \
        --disable-ld --disable-gas --disable-gdbserver --disable-sim --enable-tui \
        --disable-gdbtk --disable-shared --disable-gprofng \
        --with-expat --with-system-zlib --without-guile --with-babeltrace --with-lzma \
        --with-python=$pythonver --with-rocm-dbgapi=$ROCM_INSTALL_PATH \
        --with-amd-dbgapi PKG_CONFIG_PATH="${ROCM_INSTALL_PATH}/share/pkgconfig" \
	LDFLAGS="$LDFLAGS"
    LD_RUN_PATH='${ORIGIN}/../lib' make $MAKE_OPTS
    mkdir -p $BUILD_DIR/package/main${ROCM_INSTALL_PATH}/{share/rocgdb,bin}

    make $MAKE_OPTS -C gdb DESTDIR=$BUILD_DIR/package/main install install-pdf install-html

    make $MAKE_OPTS -C binutils DESTDIR=$BUILD_DIR/package/main install

    linkFiles $ROCM_GDB_ROOT/gdb/NOTICES.txt $BUILD_DIR/package/main${ROCM_INSTALL_PATH}/share/doc/rocgdb
    keep_wanted_files

    [ "${CPACKGEN}" = "DEB" ] || package_rpm && package_rpm_tests
    [ "${CPACKGEN}" = "RPM" ] || package_deb
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

main(){

VALID_STR=`getopt -o hcraso:p: --long help,clean,release,static,address_sanitizer,outdir:,package: -- "$@"`
eval set -- "$VALID_STR"

while true ;
do
    case "$1" in
        (-h | --help)
                printUsage ; exit 0;;
        (-c | --clean)
                TARGET="clean" ; ((CLEAN_OR_OUT|=1)) ; shift ;;
        (-r | --release)
                BUILD_TYPE="Release" ; shift ; MAKEARG="$MAKEARG REL=1" ;; # For compatability with other scripts
        (-a | --address_sanitizer)
                set_asan_env_vars
                set_address_sanitizer_on ; shift ;;
        (-s | --static)
                SHARED_LIBS="OFF" ; shift ;;
        (-o | --outdir)
                TARGET="outdir"; PKGTYPE=$2 ; OUT_DIR_SPECIFIED=1 ; ((CLEAN_OR_OUT|=2)) ; shift 2 ;;
        (-p | --package)                 #FIXME
                MAKETARGET="$2" ; shift 2;;
        # I think it would be better to use -- to indicate end of args
        # and insert an error message about unknown args at this point.
        --)     shift; break;; # end delimiter
        (*)
                echo " This should never come but just incase : UNEXPECTED ERROR Parm : [$1] ">&2 ; exit 20;;
    esac
done

if [[ $CXX == *"clang++" ]]
then
    CXX="$CXX -std=gnu++17"
fi

RET_CONFLICT=1
check_conflicting_options $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
if [ $RET_CONFLICT -ge 30 ]; then
   print_vars $API_NAME $TARGET $BUILD_TYPE $SHARED_LIBS $CLEAN_OR_OUT $PKGTYPE $MAKETARGET
   exit $RET_CONFLICT
fi

    case $TARGET in
        ("clean") clean ;;
        ("build") build ;;
        ("outdir") print_output_directory ;;
        (*) die "Invalid target $TARGET" ;;
    esac
    echo "Operation complete"
}

if [ "$0" = "$BASH_SOURCE" ]
then
    main "$@"
else
    set +e
fi
