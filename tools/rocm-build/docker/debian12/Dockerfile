FROM debian:bookworm as builder
COPY packages /tmp/packages
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN \
	perl -pi -e 's/^Components: .*/Components: main contrib non-free non-free-firmware/'  /etc/apt/sources.list.d/debian.sources && \
	groupadd debian -g 1000 && \
	useradd debian -u 1000 -g 1000 -d /home/debian && \
	apt-get update && \
	apt-get install --no-install-recommends -y $(sed 's/#.*//' /tmp/packages) && \
	apt-get clean && \
	usermod -aG sudo debian && \
	rm -rf /var/cache/apt/ /var/lib/apt/lists/* /etc/apt/apt.conf.d/01proxy
RUN \
	echo "debian ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/debian && \
	echo 'export PATH="/usr/lib/ccache:$PATH"' >> /etc/bash.bashrc && \
	pip install argparse lit CppHeaderParser rocm-docs-core --break-system-packages 
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
