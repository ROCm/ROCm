# AMD ROCm Software

ROCm is an open-source stack, composed primarily of open-source software, designed for graphics
processing unit (GPU) computation. ROCm consists of a collection of drivers, development tools, and
APIs that enable GPU programming from low-level kernel to end-user applications.

With ROCm, you can customize your GPU software to meet your specific needs. You can develop,
collaborate, test, and deploy your applications in a free, open source, integrated, and secure software
ecosystem. ROCm is particularly well-suited to GPU-accelerated high-performance computing (HPC),
artificial intelligence (AI), scientific computing, and computer aided design (CAD).

ROCm is powered by AMD’s
[Heterogeneous-computing Interface for Portability (HIP)](https://github.com/ROCm/HIP),
an open-source software C++ GPU programming environment and its corresponding runtime. HIP
allows ROCm developers to create portable applications on different platforms by deploying code on a
range of platforms, from dedicated gaming GPUs to exascale HPC clusters.

ROCm supports programming models, such as OpenMP and OpenCL, and includes all necessary open
source software compilers, debuggers, and libraries. ROCm is fully integrated into machine learning
(ML) frameworks, such as PyTorch and TensorFlow.

## Getting the ROCm Source Code

AMD ROCm is built from open source software. It is, therefore, possible to modify the various components of ROCm by downloading the source code and rebuilding the components. The source code for ROCm components can be cloned from each of the GitHub repositories using git.  For easy access to download the correct versions of each of these tools, the ROCm repository contains a repo manifest file called [default.xml](./default.xml). You can use this manifest file to download the source code for ROCm software.

### Installing the repo tool

The repo tool from Google allows you to manage multiple git repositories simultaneously. Run the following commands to install the repo tool:

```bash
mkdir -p ~/bin/
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
```

**Note:** The ```~/bin/``` folder is used as an example. You can specify a different folder to install the repo tool into if you desire.

### Installing git-lfs

Some ROCm projects use the Git Large File Storage (LFS) format that may require you to install git-lfs. Refer to [Git Large File Storage](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md) for more information. For example, to install git-lfs for Ubuntu, use the following command:

```bash
sudo apt-get install git-lfs
```

### Downloading the ROCm source code

The following example shows how to use the repo tool to download the ROCm source code. If you choose a directory other than ~/bin/ to install the repo tool, you must use that chosen directory in the code as shown below:

```bash
mkdir -p ~/ROCm/
cd ~/ROCm/
~/bin/repo init -u http://github.com/ROCm/ROCm.git -b roc-6.3.x
~/bin/repo sync
```

**Note:** Using this sample code will cause the repo tool to download the open source code associated with the specified ROCm release. Ensure that you have ssh-keys configured on your machine for your GitHub ID prior to the download as explained at [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

## Building the ROCm source code

Each ROCm component repository contains directions for building that component, such as the rocSPARSE documentation [Installation and Building for Linux](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/install/Linux_Install_Guide.html). Refer to the specific component documentation for instructions on building the repository.

Each release of the ROCm software supports specific hardware and software configurations. Refer to [System requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for the current supported hardware and OS.

## Build ROCm from source

The Build will use as many processors as it can find to build in parallel. Some of the compiles can consume as much as 10GB of RAM, so make sure you have plenty of Swap Space !

By default the ROCm build will compile for all supported GPU architectures and will take approximately 500 CPU hours.
The Build time will reduce significantly if we limit the GPU Architecture/s against which we need to build by using the environment variable GPU_ARCHS as mentioned below.

```bash
# --------------------------------------
# Step1: clone source code
# --------------------------------------

mkdir -p ~/WORKSPACE/      # Or any folder name other than WORKSPACE
cd ~/WORKSPACE/
export ROCM_VERSION=6.3.1
~/bin/repo init -u http://github.com/ROCm/ROCm.git -b roc-6.3.x -m tools/rocm-build/rocm-${ROCM_VERSION}.xml
~/bin/repo sync

# --------------------------------------
# Step 2: Prepare build environment
# --------------------------------------

# Option 1: Start a docker container
# Pulling required base docker images:
# Ubuntu20.04 built from ROCm/tools/rocm-build/docker/ubuntu20/Dockerfile
docker pull rocm/rocm-build-ubuntu-20.04:6.2
# Ubuntu22.04 built from ROCm/tools/rocm-build/docker/ubuntu22/Dockerfile
docker pull rocm/rocm-build-ubuntu-22.04:6.2
# Ubuntu24.04 built from ROCm/tools/rocm-build/docker/ubuntu24/Dockerfile
docker pull rocm/rocm-build-ubuntu-24.04:6.2

# Start docker container and mount the source code folder:
docker run -ti \
    -e ROCM_VERSION=${ROCM_VERSION} \
    -e CCACHE_DIR=$HOME/.ccache \
    -e CCACHE_ENABLED=true \
    -e DOCK_WORK_FOLD=/src \
    -w /src \
    -v $PWD:/src \
    -v /etc/passwd:/etc/passwd \
    -v /etc/shadow:/etc/shadow \
    -v ${HOME}/.ccache:${HOME}/.ccache \
    -u $(id -u):$(id -g) \
    <replace_with_required_ubuntu_base_docker_image> bash

# Option 2: Install required packages into the host machine
# For ubuntu20.04 system
cd ROCm/tools/rocm-build/docker/ubuntu20
cp * /tmp && cd /tmp
bash install-prerequisites.sh
# For ubuntu22.04 system
cd ROCm/tools/rocm-build/docker/ubuntu22
cp * /tmp && cd /tmp
bash install-prerequisities.sh
# For ubuntu24.04 system
cd ROCm/tools/rocm-build/docker/ubuntu24
cp * /tmp && cd /tmp
bash install-prerequisites.sh

# --------------------------------------
# Step 3: Run build command line
# --------------------------------------

# Select GPU targets before building:
# When GPU_ARCHS is not set, default GPU targets supported by ROCm6.1 will be used.
# To build against a subset of GFX architectures you can use the below env variable.
# Support MI300 (gfx940, gfx941, gfx942).
export GPU_ARCHS="gfx942"               # Example
export GPU_ARCHS="gfx940;gfx941;gfx942" # Example

# Pick and run build commands in the docker container:
# Build rocm-dev packages
make -f ROCm/tools/rocm-build/ROCm.mk -j ${NPROC:-$(nproc)} rocm-dev
# Build all ROCm packages
make -f ROCm/tools/rocm-build/ROCm.mk -j ${NPROC:-$(nproc)} all
# list all ROCm components to find required components
make -f ROCm/tools/rocm-build/ROCm.mk list_components
# Build a single ROCm packages
make -f ROCm/tools/rocm-build/ROCm.mk T_rocblas

# Find built packages in ubuntu20.04:
out/ubuntu-20.04/20.04/deb/
# Find built packages in ubuntu22.04:
out/ubuntu-22.04/22.04/deb/
# Find built packages in ubuntu24.04:
out/ubuntu-24.04/24.04/deb/

# Find built logs in ubuntu20.04:
out/ubuntu-20.04/20.04/logs/
# Find built logs in ubuntu22.04:
out/ubuntu-22.04/22.04/logs/
# Find built logs in ubuntu24.04:
out/ubuntu-24.04/24.04/logs/
# All logs pertaining to failed components, end with .errrors extension.
out/ubuntu-22.04/22.04/logs/rocblas.errors      # Example
# All logs pertaining to building components, end with .inprogress extension.
out/ubuntu-22.04/22.04/logs/rocblas.inprogress  # Example
# All logs pertaining to passed components, use the component names.
out/ubuntu-22.04/22.04/logs/rocblas             # Example
```

Note: [Overview for ROCm.mk](tools/rocm-build/README.md)

## ROCm documentation

This repository contains the [manifest file](https://gerrit.googlesource.com/git-repo/+/HEAD/docs/manifest-format.md)
for ROCm releases, changelogs, and release information.

The `default.xml` file contains information for all repositories and the associated commit used to build
the current ROCm release; `default.xml` uses the [Manifest Format repository](https://gerrit.googlesource.com/git-repo/).

Source code for our documentation is located in the `/docs` folder of most ROCm repositories. The
`develop` branch of our repositories contains content for the next ROCm release.

The ROCm documentation homepage is [rocm.docs.amd.com](https://rocm.docs.amd.com).

For information on how to contribute to the ROCm documentation, see [Contributing to the ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Older ROCm releases

For release information for older ROCm releases, refer to the
[ROCm release history](https://rocm.docs.amd.com/en/latest/release/versions.html).
