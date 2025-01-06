#! /usr/bin/bash

set -ex

apt-get -y update
DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true apt-get install --no-install-recommends -y $(sed 's/#.*//' /tmp/packages)
apt-get clean
rm -rf /var/cache/apt/ /var/lib/apt/lists/* /etc/apt/apt.conf.d/01proxy

#Install git 2.17.1 version
curl -o git.tar.gz https://cdn.kernel.org/pub/software/scm/git/git-2.17.1.tar.gz
tar -zxf git.tar.gz
cd git-*
make prefix=/usr/local all
make prefix=/usr/local install
git --version

pip3 install --no-cache-dir setuptools wheel tox
pip3 install --no-cache-dir CppHeaderParser argparse requests lxml barectf recommonmark jinja2==3.0.0 \
    websockets matplotlib numpy==1.26.4 scipy minimal msgpack pytest sphinx joblib PyYAML rocm-docs-core cmake==3.25.2 pandas myst-parser lit

# Allow sudo for everyone user
echo 'ALL ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/everyone

wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
apt install -y ./amdgpu-install_6.3.60300-1_all.deb
apt update
apt install -y libva-amdgpu-dev libdrm-amdgpu-dev
rm -rf /etc/apt/sources.list.d/amdgpu.list /etc/apt/sources.list.d/amdgpu-proprietary.list ./amdgpu-install_6.3.60300-1_all.deb
apt autoremove -y --purge amdgpu-install
apt update

# Build ccache from the source
cd /tmp
git clone https://github.com/ccache/ccache -b v4.7.5
cd ccache
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
cd /tmp
rm -rf ccache

# symlinks for ccache
ln -s /usr/local/bin/ccache /usr/local/bin/gcc
ln -s /usr/local/bin/ccache /usr/local/bin/g++
ln -s /usr/local/bin/ccache /usr/local/bin/cc
ln -s /usr/local/bin/ccache /usr/local/bin/c++
ln -s /usr/local/bin/ccache /usr/local/bin/amdclang
ln -s /usr/local/bin/ccache /usr/local/bin/amdclang++
ln -s /usr/local/bin/ccache /usr/local/bin/hipcc

# Install OCaml packages
wget -nv https://sourceforge.net/projects/opam.mirror/files/2.1.6/opam-2.1.6-x86_64-linux/download -O /usr/local/bin/opam
chmod +x /usr/local/bin/opam
opam init --yes --disable-sandboxing
opam install ctypes --yes

# Install and modify git-repo
curl https://storage.googleapis.com/git-repo-downloads/repo > /usr/bin/repo
chmod a+x /usr/bin/repo

cd /var/tmp
mkdir mlnx
wget -O mlnx/tar.tgz https://content.mellanox.com/ofed/MLNX_OFED-24.01-0.3.3.1/MLNX_OFED_LINUX-24.01-0.3.3.1-ubuntu22.04-x86_64.tgz
tar -xz -C mlnx -f mlnx/tar.tgz
apt-key add mlnx/*/RPM-GPG-KEY-Mellanox
echo "deb [arch=amd64] file:$(echo $PWD/mlnx/*/DEBS) ./" > /etc/apt/sources.list.d/sharp.list
apt update
apt install -y sharp
apt clean
rm -rf /var/cache/apt/ /var/lib/apt/lists/* mlnx /etc/apt/sources.list.d/sharp.list

apt update
apt -y install libunwind-dev
apt -y install libgoogle-glog-dev

# Install python3.8 from source
curl -LO https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tar.xz
tar -xvf Python-3.8.13.tar.xz
pwd
ls /var/tmp/
ls Python-3.8.13
mv Python-3.8.13 /opt/
apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev pkg-config make -y
cd /opt/Python-3.8.13/
./configure --enable-optimizations --enable-shared
make
make -j 6
make altinstall
ldconfig /opt/Python3.8.13
python3.8 --version

python3.8 -m pip install setuptools wheel
python3.8 -m pip install CppHeaderParser argparse requests lxml PyYAML joblib

curl -lO https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.13.tar.bz2
tar -xvf hwloc-1.11.13.tar.bz2
cd hwloc-1.11.13
./configure
make
make install
cp /usr/local/lib/libhwloc.so.5 /usr/lib
hwloc-info --version

# Install googletest from source
mkdir -p /tmp/gtest
cd /tmp/gtest
wget https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip -O googletest.zip
unzip googletest.zip
cd googletest-1.14.0/
mkdir build
cd build
cmake ..
make -j$(nproc)
make install
rm -rf /tmp/gtest

# Install gRPC from source
GRPC_ARCHIVE=grpc-1.61.0.tar.gz
mkdir /tmp/grpc
mkdir /usr/grpc
cd /tmp
git clone --recurse-submodules -b v1.61.0 https://github.com/grpc/grpc
cd grpc
mkdir -p build
cd build
cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/grpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 -DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,--enable-new-dtags,--build-id=sha1,--rpath,'$ORIGIN' ..
make -j $(nproc) install
rm -rf /tmp/grpc

# Download prebuilt AMD multithreaded blis (2.0)
mkdir -p /tmp/blis
cd /tmp/blis
wget -O - https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz | tar xfz -
mv amd-blis-mt /usr/blis
cd /
rm -rf /tmp/blis

# Download aocl-linux-gcc-4.2.0_1_amd64.deb
mkdir -p /tmp/aocl
cd /tmp/aocl
wget -nv https://download.amd.com/developer/eula/aocl/aocl-4-2/aocl-linux-gcc-4.2.0_1_amd64.deb
apt install ./aocl-linux-gcc-4.2.0_1_amd64.deb
rm -rf /tmp/aocl

# lapack(3.9.1v)
lapack_version=3.9.1
lapack_srcdir=lapack-$lapack_version
lapack_blddir=lapack-$lapack_version-bld
mkdir -p /tmp/lapack
cd /tmp/lapack
rm -rf "$lapack_srcdir" "$lapack_blddir"
wget -O - https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.9.1.tar.gz | tar xzf -
cmake -H$lapack_srcdir -B$lapack_blddir -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_FLAGS=-fno-optimize-sibling-calls -DBUILD_TESTING=OFF -DCBLAS=ON -DLAPACKE=OFF
make -j$(nproc) -C "$lapack_blddir"
make -C "$lapack_blddir" install
cd $lapack_blddir
cp -r ./include/* /usr/local/include/
cp -r ./lib/* /usr/local/lib
cd /
rm -rf /tmp/lapack

# FMT(7.1.3v)
fmt_version=7.1.3
fmt_srcdir=fmt-$fmt_version
fmt_blddir=fmt-$fmt_version-bld
mkdir -p /tmp/fmt
cd /tmp/fmt
rm -rf "$fmt_srcdir" "$fmt_blddir"
wget -O - https://github.com/fmtlib/fmt/archive/refs/tags/7.1.3.tar.gz | tar xzf -
cmake -H$fmt_srcdir -B$fmt_blddir -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_STANDARD_REQUIRED=ON -DFMT_DOC=OFF -DFMT_TEST=OFF
make -j$(nproc) -C "$fmt_blddir"
make -C "$fmt_blddir" install

# Build and install libjpeg-turbo
mkdir -p /tmp/libjpeg-turbo
cd /tmp/libjpeg-turbo
wget -nv https://github.com/rrawther/libjpeg-turbo/archive/refs/heads/2.0.6.2.zip -O libjpeg-turbo-2.0.6.2.zip
unzip libjpeg-turbo-2.0.6.2.zip
cd libjpeg-turbo-2.0.6.2
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ..
make -j$(nproc) install
rm -rf /tmp/libjpeg-turbo

# Get released ninja from source
mkdir -p /tmp/ninja
cd /tmp/ninja
wget -nv https://codeload.github.com/Kitware/ninja/zip/refs/tags/v1.11.1.g95dee.kitware.jobserver-1 -O ninja.zip
unzip ninja.zip
cd ninja-1.11.1.g95dee.kitware.jobserver-1
./configure.py --bootstrap
cp ninja /usr/local/bin/
rm -rf /tmp/ninja

# Build NASM
mkdir -p /tmp/nasm-2.15.05
cd /tmp
wget -qO- "https://distfiles.macports.org/nasm/nasm-2.15.05.tar.bz2" | tar -xvj
cd nasm-2.15.05
./autogen.sh
./configure --prefix="/usr/local"
make -j$(nproc) install
rm -rf /tmp/nasm-2.15.05

# Build YASM
mkdir -p /tmp/yasm-1.3.0
cd /tmp
wget -qO- "http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz" | tar -xvz
cd yasm-1.3.0
./configure --prefix="/usr/local"
make -j$(nproc) install
rm -rf /tmp/yasm-1.3.0

# Build x264
mkdir -p /tmp/x264-snapshot-20191217-2245-stable
cd /tmp
wget -qO-  "https://download.videolan.org/pub/videolan/x264/snapshots/x264-snapshot-20191217-2245-stable.tar.bz2" | tar -xvj
cd /tmp/x264-snapshot-20191217-2245-stable
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure --prefix="/usr/local" --enable-shared
make -j$(nproc) install
rm -rf /tmp/x264-snapshot-20191217-2245-stable

# Build x265
mkdir -p /tmp/x265_2.7
cd /tmp
wget -qO- "https://get.videolan.org/x265/x265_2.7.tar.gz" | tar -xvz
cd  /tmp/x265_2.7/build/linux
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/usr/local" -DENABLE_SHARED:bool=on ../../source
make -j$(nproc) install
rm -rf /tmp/x265_2.7

# Build fdk-aac
mkdir -p /tmp/fdk-aac-2.0.2
cd /tmp
wget -qO- "https://sourceforge.net/projects/opencore-amr/files/fdk-aac/fdk-aac-2.0.2.tar.gz" | tar -xvz
cd /tmp/fdk-aac-2.0.2
autoreconf -fiv
./configure --prefix="/usr/local" --enable-shared --disable-static
make -j$(nproc) install
rm -rf /tmp/fdk-aac-2.0.2

# Build FFmpeg
cd /tmp
git clone -b release/4.4 https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"
./configure  --prefix="/usr/local" --extra-cflags="-I/usr/local/include"   --extra-ldflags="-L/usr/local/lib"  --extra-libs=-lpthread  --extra-libs=-lm  --enable-shared   --disable-static   --enable-libx264  --enable-libx265  --enable-libfdk-aac  --enable-gpl --enable-nonfree
make -j$(nproc) install
rm -rf /tmp/ffmpeg

cp /tmp/local-pin-600 /etc/apt/preferences.d

command -v lbzip2
ln -sf $(command -v lbzip2) /usr/local/bin/compressor || ln -sf $(command -v bzip2) /usr/local/bin/compressor

# Install Google Benchmark
mkdir -p /tmp/Gbenchmark
cd /tmp/Gbenchmark
wget -qO- https://github.com/google/benchmark/archive/refs/tags/v1.6.1.tar.gz | tar xz
cmake -Sbenchmark-1.6.1 -Bbuild -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_CXX_STANDARD=14
make -j -C build
cd /tmp/Gbenchmark/build
make install

mkdir -p /tmp/boost-1.85.0
cd /tmp/boost-1.85.0
wget -nv https://sourceforge.net/projects/boost/files/boost/1.85.0/boost_1_85_0.tar.bz2 -O ./boost_1_85_0.tar.bz2
tar -xf boost_1_85_0.tar.bz2 --use-compress-program="/usr/local/bin/compressor"
cd boost_1_85_0
./bootstrap.sh --prefix=${RPP_DEPS_LOCATION} --with-python=python3
./b2 stage -j$(nproc) threading=multi link=shared cxxflags="-std=c++11"
./b2 install threading=multi link=shared --with-system --with-filesystem
./b2 stage -j$(nproc) threading=multi link=static cxxflags="-std=c++11 -fpic" cflags="-fpic"
./b2 install threading=multi link=static --with-system --with-filesystem
rm -rf /tmp/boost-1.85.0
