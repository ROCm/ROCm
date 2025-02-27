# lapack(3.9.1v)
set -e
lapack_version=3.9.1
#lapack_version=3.12.1
lapack_srcdir=lapack-$lapack_version
lapack_blddir=lapack-$lapack_version-bld
mkdir -p /tmp/lapack
cd /tmp/lapack
rm -rf "$lapack_srcdir" "$lapack_blddir"
wget -O - https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${lapack_version}.tar.gz | tar xzf -
export CC=/usr/lib/ccache/gcc-14  
cmake -H$lapack_srcdir -B$lapack_blddir -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_FLAGS=-fno-optimize-sibling-calls -DBUILD_TESTING=OFF -DCBLAS=ON -DLAPACKE=OFF
make -j$(nproc) -C "$lapack_blddir"
make -C "$lapack_blddir" install
cd $lapack_blddir
cp -r ./include/* /usr/local/include/
cp -r ./lib/* /usr/local/lib
cd /
rm -rf /tmp/lapack
