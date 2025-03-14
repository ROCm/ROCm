# Build and install libjpeg-turbo
mkdir -p /tmp/libjpeg-turbo
cd /tmp/libjpeg-turbo
#wget -nv https://github.com/rrawther/libjpeg-turbo/archive/refs/heads/2.0.6.2.zip -O libjpeg-turbo-2.0.6.2.zip
wget -nv https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.0.zip -O libjpeg-turbo-3.1.0.zip
#unzip libjpeg-turbo-2.0.6.2.zip
unzip libjpeg-turbo-3.1.0.zip
#cd libjpeg-turbo-2.0.6.2
cd libjpeg-turbo-3.1.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ..
make -j$(nproc) install
rm -rf /tmp/libjpeg-turbo
