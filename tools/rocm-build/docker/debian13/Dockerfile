FROM debian:trixie as builder
COPY packages /tmp/packages
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN \
	perl -pi -e 's/^Components: .*/Components: main contrib non-free/'  /etc/apt/sources.list.d/debian.sources && \
	perl -pi -e 's/^Suites: trixie trixie.*/Suites: trixie trixie-updates bookworm bookworm-updates/'  /etc/apt/sources.list.d/debian.sources && \
	perl -pi -e 's/^Suites: trixie-sec.*/Suites: trixie-security bookworm-security/'  /etc/apt/sources.list.d/debian.sources && \
	groupadd debian -g 1000 && \
	useradd debian -u 1000 -g 1000 -d /home/debian && \
	apt-get update && \
	apt-get install --no-install-recommends -y $(sed 's/#.*//' /tmp/packages) && \
	apt-get clean && \
	usermod -aG sudo debian && \
	echo "debian ALL=NOPASSWD: /usr/bin/apt,/usr/bin/make" > /etc/sudoers.d/debian && \
	rm -rf /var/cache/apt/ /var/lib/apt/lists/* /etc/apt/apt.conf.d/01proxy
RUN \
	grep -q VERSION_ID /etc/os-release || echo "VERSION_ID=13" >> /etc/os-release && \
	echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && \
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 10 && \
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 10 && \
	echo 'export PATH="/usr/lib/ccache:$PATH"' >> /etc/bash.bashrc && \
	pip install argparse lit CppHeaderParser rocm-docs-core --break-system-packages
#COPY python/argparse/python3-argparse_1.4.0-1_all.deb /tmp
#COPY python/lit/python3-lit_18.1.8-1_all.deb /tmp
#COPY python/rocm-docs-core/python3-rocm-docs-core_1.15.0-1_all.deb /tmp
#COPY python/CppHeaderParser/python3-cppheaderparser_2.7.4-1_all.deb /tmp
#RUN  apt update && apt install --no-install-recommends -y /tmp/*.deb && apt-get clean
# Build lapack with fortran support for hipblaslt
COPY build_lapack.sh /tmp
RUN /tmp/build_lapack.sh
# Build grpc 1.61 for rdc
COPY build_grpc.sh /tmp
RUN /tmp/build_grpc.sh
# Build turbojpeg for rocAL
COPY build_turbojpeg.sh /tmp
RUN /tmp/build_turbojpeg.sh
RUN /usr/sbin/update-ccache-symlinks
	
