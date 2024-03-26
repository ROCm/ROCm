# What is ROCm?

ROCm is an open-source stack for GPU computation. ROCm is primarily Open-Source
Software (OSS) that allows developers the freedom to customize and tailor their
GPU software for their own needs while collaborating with a community of other
developers, and helping each other find solutions in an agile, flexible, rapid
and secure manner.

ROCm is a collection of drivers, development tools and APIs enabling GPU
programming from the low-level kernel to end-user applications. ROCm is powered
by AMD’s Heterogeneous-computing Interface for Portability (HIP), an OSS C++ GPU
programming environment and its corresponding runtime. HIP allows ROCm
developers to create portable applications on different platforms by deploying
code on a range of platforms, from dedicated gaming GPUs to exascale HPC
clusters. ROCm supports programming models such as OpenMP and OpenCL, and
includes all the necessary OSS compilers, debuggers and libraries. ROCm is fully
integrated into ML frameworks such as PyTorch and TensorFlow. ROCm can be
deployed in many ways, including through the use of containers such as Docker,
Spack, and your own build from source.

The goal of ROCm is to allow our users to maximize their GPU hardware investment.
ROCm is designed to help develop, test and deploy GPU accelerated HPC, AI,
scientific computing, CAD, and other applications in a free, open-source,
integrated and secure software ecosystem.

## ROCm on Windows

Starting with ROCm 5.5, the HIP SDK brings a subset of ROCm to developers on Windows.
The collection of features enabled on Windows is referred to as the HIP SDK.
These features allow developers to use the HIP runtime, HIP math libraries
and HIP Primitive libraries. The following table shows the differences
between Windows and Linux releases.

|Component|Linux|Windows|
|---------|-----|-------|
|Driver|Radeon Software for Linux |AMD Software Pro Edition|
|Compiler|`hipcc`/`amdclang++`|`hipcc`/`clang++`|
|Debugger|`rocgdb`|no debugger available|
|Profiler|`rocprof`|[Radeon GPU Profiler](https://gpuopen.com/rgp/)|
|Porting Tools|HIPIFY|Coming Soon|
|Runtime|HIP (Open Sourced)|HIP (closed source)|
|Math Libraries|Supported|Supported|
|Primitives Libraries|Supported|Supported|
|Communication Libraries|Supported|Not Available|
|AI Libraries|MIOpen, MIGraphX|Not Available|
|System Management|`rocm-smi-lib`, RDC, `rocminfo`|`amdsmi`, `hipInfo`|
|AI Frameworks|PyTorch, TensorFlow, etc.|Not Available|
|CMake HIP Language|Enabled|Unsupported|
|Visual Studio| Not applicable| Plugin Available|
|HIP Ray Tracing| Supported|Supported|

AMD is continuing to invest in Windows support and AMD plans to release enhanced
features in subsequent revisions.

```{note}
The 5.5 Windows Installer collectively groups the Math and Primitives
libraries.
```

```{note}
GPU support on Windows and Linux may differ. You must refer to
Windows and Linux GPU support tables separately.
```

```{note}
HIP Ray Tracing is not distributed via ROCm in Linux.
```

### ROCm release versioning

Linux OS releases set the canonical version numbers for ROCm. Windows will
follow Linux version numbers as Windows releases are based on Linux ROCm
releases. However, not all Linux ROCm releases will have a corresponding Windows
release. The following table shows the ROCm releases on Windows and Linux. Releases
with both Windows and Linux are referred to as a joint release. Releases with
only Linux support are referred to as a skipped release from the Windows
perspective.

|Release version|Linux|Windows|
|---------------|-----|-------|
|5.5|✅|✅|
|5.6|✅|❌|

ROCm Linux releases are versioned with following the Major.Minor.Patch
version number system. Windows releases will only be versioned with Major.Minor.

In general, Windows releases will trail Linux releases. Software developers that
wish to support both Linux and Windows using a single ROCm version should
refrain from upgrading ROCm unless there is a joint release.

### Windows Documentation implications

The ROCm documentation website contains both Windows and Linux documentation.
Just below each article title, a convenient article information section states
whether the page applies to Linux only, Windows only or both OSes. To find the
exact Windows documentation for a release of the HIP SDK, please view the ROCm documentation with the same
Major.Minor version number while ignoring the Patch version. The Patch version
only matters for Linux releases.  For convenience,
Windows documentation will continue to be included in the overall ROCm
documentation for the skipped Windows releases.

Windows release notes will contain only information pertinent to Windows.
The software developer must read all the previous ROCm release notes (including)
skipped ROCm versions on Windows for information on all the changes present in
the Windows release.

### Windows Builds from Source

Not all source code required to build Windows from source is available under a
permissive open source license. Build instructions on Windows is only provided
for projects that can be built from source on Windows using a toolchain that
has closed source build prerequisites. The ROCm manifest file is not valid for
Windows. AMD does not release a manifest or tag our components in Windows.
Users may use corresponding Linux tags to build on Windows.
