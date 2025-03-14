build:
	docker build -t rocm/rocm-build-debian-13 .

fullbuild:
	docker build --no-cache -t rocm/rocm-build-debian-13 .

run:
	docker run --rm -it --device /dev/kfd --device /dev/dri \
		-e CCACHE_DIR=/home/debian/.ccache \
		-e CCACHE_ENABLED=true \
		-e DOCK_WORK_FOLD=/src \
		--security-opt seccomp=unconfined \
		--group-add $(shell getent group video | cut -d: -f3) \
		--group-add $(shell getent group render | cut -d: -f3) \
		-v ${HOME}/.ccache:/home/debian/.ccache \
		-v $(CURDIR)/../..:/src \
		--user=1000:1000 \
		-w /src \
		--name rocm-build-debian-13 \
		rocm/rocm-build-debian-13:latest \
		/bin/bash
