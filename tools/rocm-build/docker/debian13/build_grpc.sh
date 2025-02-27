# Install gRPC from source
GRPC_ARCHIVE=grpc-1.61.0.tar.gz
mkdir /tmp/grpc
mkdir /usr/grpc
cd /tmp
git clone --recurse-submodules -b v1.61.0 https://github.com/grpc/grpc
cd grpc
mkdir -p build 
cd build
cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/grpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 ..
make -j $(nproc) install
rm -rf /tmp/grpc
