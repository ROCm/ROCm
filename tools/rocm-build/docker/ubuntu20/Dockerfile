FROM ubuntu:focal as builder
COPY packages /tmp/packages
COPY install-prerequisites.sh /tmp/install-prerequisites.sh
RUN chmod +x /tmp/install-prerequisites.sh
ENV KBUILD_PKG_ROOTCMD=
ENV RPP_DEPS_LOCATION=/usr/local/rpp-deps
RUN /tmp/install-prerequisites.sh
WORKDIR /src
