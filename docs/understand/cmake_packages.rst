***********
Using CMake
***********

Most components in ROCm support CMake. Projects depending on header-only or
library components typically require CMake 3.5 or higher whereas those wanting
to make use of the CMake HIP language support will require CMake 3.21 or higher.

Finding Dependencies
====================

.. note::
   For a complete
   reference on how to deal with dependencies in CMake, refer to the CMake docs
   on `find_package
   <https://cmake.org/cmake/help/latest/command/find_package.html>`_ and the
   `Using Dependencies Guide
   <https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html>`_
   to get an overview of CMake related facilities.

In short, CMake supports finding dependencies in two ways:

-  In Module mode, it consults a file ``Find<PackageName>.cmake`` which tries to
   find the component in typical install locations and layouts. CMake ships a
   few dozen such scripts, but users and projects may ship them as well.
-  In Config mode, it locates a file named ``<packagename>-config.cmake`` or
   ``<PackageName>Config.cmake`` which describes the installed component in all
   regards needed to consume it.

ROCm predominantly relies on Config mode, one notable exception being the Module
driving the compilation of HIP programs on NVIDIA runtimes. As such, when
dependencies are not found in standard system locations, one either has to
instruct CMake to search for package config files in additional folders using
the ``CMAKE_PREFIX_PATH`` variable (a semi-colon separated list of filesystem
paths), or using ``<PackageName>_ROOT`` variable on a project-specific basis.

There are nearly a dozen ways to set these variables. One may be more convenient
over the other depending on your workflow. Conceptually the simplest is adding
it to your CMake configuration command on the command-line via
``-D CMAKE_PREFIX_PATH=....`` . AMD packaged ROCm installs can typically be
added to the config file search paths such as:

-  Windows: ``-D CMAKE_PREFIX_PATH=${env:HIP_PATH}``

-  Linux: ``-D CMAKE_PREFIX_PATH=/opt/rocm``

ROCm provides the respective *config-file* packages, and this enables
``find_package`` to be used directly. ROCm does not require any Find module as
the *config-file* packages are shipped with the upstream projects, such as
rocPRIM and other ROCm libraries.

For a complete guide on where and how ROCm may be installed on a system, refer
to the installation guides in these docs (`Linux <../deploy/linux/index.html>`_).

Using HIP in CMake
==================

ROCm components providing a C/C++ interface support being consumed using any
C/C++ toolchain that CMake knows how to drive. ROCm also supports the CMake HIP
language features, allowing users to program using the HIP single-source
programming model. When a program (or translation-unit) uses the HIP API without
compiling any GPU device code, HIP can be treated in CMake as a simple C/C++
library.

Using the HIP single-source programming model
---------------------------------------------

Source code written in the HIP dialect of C++ typically uses the `.hip`
extension. When the HIP CMake language is enabled, it will automatically
associate such source files with the HIP toolchain being used.

::

    cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
    cmake_policy(VERSION 3.21.3...3.27)
    project(MyProj LANGUAGES HIP)
    add_executable(MyApp Main.hip)

Should you have existing CUDA code that is from the source compatible subset of
HIP, you can tell CMake that despite their `.cu` extension, they're HIP sources.
Do note that this mostly facilitates compiling kernel code-only source files,
as host-side CUDA API won't compile in this fashion.

::

    add_library(MyLib MyLib.cu)
    set_source_files_properties(MyLib.cu PROPERTIES LANGUAGE HIP)

CMake itself only hosts part of the HIP language support, such as defining
HIP-specific properties, etc. while the other half ships with the HIP
implementation, such as ROCm. CMake will search for a file
`hip-lang-config.cmake` describing how the the properties defined by CMake
translate to toolchain invocations. If one installs ROCm using non-standard
methods or layouts and CMake can't locate this file or detect parts of the SDK,
there's a catch-all, last resort variable consulted locating this file,
``-D CMAKE_HIP_COMPILER_ROCM_ROOT:PATH=`` which should be set the root of the
ROCm installation.

If the user doesn't provide a semi-colon delimited list of device architectures
via ``CMAKE_HIP_ARCHITECTURES``, CMake will select some sensible default. It is
advised though that if a user knows what devices they wish to target, then set
this variable explicitly.

Consuming ROCm C/C++ Libraries
------------------------------

Libraries such as rocBLAS, rocFFT, MIOpen, etc. behave as C/C++ libraries.
Illustrated in the example below is a C++ application using MIOpen from CMake.
It calls ``find_package(miopen)``, which provides the ``MIOpen`` imported
target. This can be linked with ``target_link_libraries``

::

    cmake_minimum_required(VERSION 3.5) # find_package(miopen) requires 3.5
    cmake_policy(VERSION 3.5...3.27)
    project(MyProj LANGUAGES CXX)
    find_package(miopen)
    add_library(MyLib ...)
    target_link_libraries(MyLib PUBLIC MIOpen)

.. note::
    Most libraries are designed as host-only API, so using a GPU device
    compiler is not necessary for downstream projects unless they use GPU device
    code.

Consuming the HIP API in C++ code
---------------------------------

Use the HIP API without compiling the GPU device code. As there is no GPU code,
any C or C++ compiler can be used. The ``find_package(hip)`` provides the
``hip::host`` imported target to use HIP in this context.

::

    cmake_minimum_required(VERSION 3.5) # find_package(hip) requires 3.5
    cmake_policy(VERSION 3.5...3.27)
    project(MyProj LANGUAGES CXX)
    find_package(hip REQUIRED)
    add_executable(MyApp ...)
    target_link_libraries(MyApp PRIVATE hip::host)

Compiling device code in C++ language mode
------------------------------------------

.. attention::
    The workflow detailed here is considered legacy and is shown for
    understanding's sake. It pre-dates the existence of HIP language support in
    CMake. If source code has HIP device code in it, it is a HIP source file
    and should be compiled as such. Only resort to the method below if your
    HIP-enabled CMake codepath can't mandate CMake version 3.21.

If code uses the HIP API and compiles GPU device code, it requires using a
device compiler. The compiler for CMake can be set using either the
``CMAKE_C_COMPILER`` and ``CMAKE_CXX_COMPILER`` variable or using the ``CC``
and ``CXX`` environment variables. This can be set when configuring CMake or
put into a CMake toolchain file. The device compiler must be set to a
compiler that supports AMD GPU targets, which is usually Clang.

The ``find_package(hip)`` provides the ``hip::device`` imported target to add
all the flags necessary for device compilation.

::

    cmake_minimum_required(VERSION 3.8) # cxx_std_11 requires 3.8
    cmake_policy(VERSION 3.8...3.27)
    project(MyProj LANGUAGES CXX)
    find_package(hip REQUIRED)
    add_library(MyLib ...)
    target_link_libraries(MyLib PRIVATE hip::device)
    target_compile_features(MyLib PRIVATE cxx_std_11)

.. note::
    Compiling for the GPU device requires at least C++11.

This project can then be configured with the following CMake commands:

-  Windows: ``cmake -D CMAKE_CXX_COMPILER:PATH=${env:HIP_PATH}\bin\clang++.exe``

-  Linux: ``cmake -D CMAKE_CXX_COMPILER:PATH=/opt/rocm/bin/amdclang++``

Which use the device compiler provided from the binary packages of
`ROCm HIP SDK <https://www.amd.com/en/developer/rocm-hub.html>`_ and
`repo.radeon.com <https://repo.radeon.com>`_ respectively.

When using the CXX language support to compile HIP device code, selecting the
target GPU architectures is done via setting the ``GPU_TARGETS`` variable.
``CMAKE_HIP_ARCHITECTURES`` only exists when the HIP language is enabled. By
default, this is set to some subset of the currently supported architectures of
AMD ROCm. It can be set to the CMake option ``-D GPU_TARGETS="gfx1032;gfx1035"``.

ROCm CMake Packages
-------------------

+-----------+----------+--------------------------------------------------------+
| Component | Package  | Targets                                                |
+===========+==========+========================================================+
| HIP       | hip      | ``hip::host``, ``hip::device``                         |
+-----------+----------+--------------------------------------------------------+
| rocPRIM   | rocprim  | ``roc::rocprim``                                       |
+-----------+----------+--------------------------------------------------------+
| rocThrust | rocthrust| ``roc::rocthrust``                                     |
+-----------+----------+--------------------------------------------------------+
| hipCUB    | hipcub   | ``hip::hipcub``                                        |
+-----------+----------+--------------------------------------------------------+
| rocRAND   | rocrand  | ``roc::rocrand``                                       |
+-----------+----------+--------------------------------------------------------+
| rocBLAS   | rocblas  | ``roc::rocblas``                                       |
+-----------+----------+--------------------------------------------------------+
| rocSOLVER | rocsolver| ``roc::rocsolver``                                     |
+-----------+----------+--------------------------------------------------------+
| hipBLAS   | hipblas  | ``roc::hipblas``                                       |
+-----------+----------+--------------------------------------------------------+
| rocFFT    | rocfft   | ``roc::rocfft``                                        |
+-----------+----------+--------------------------------------------------------+
| hipFFT    | hipfft   | ``hip::hipfft``                                        |
+-----------+----------+--------------------------------------------------------+
| rocSPARSE | rocsparse| ``roc::rocsparse``                                     |
+-----------+----------+--------------------------------------------------------+
| hipSPARSE | hipsparse| ``roc::hipsparse``                                     |
+-----------+----------+--------------------------------------------------------+
| rocALUTION|rocalution| ``roc::rocalution``                                    |
+-----------+----------+--------------------------------------------------------+
| RCCL      | rccl     | ``rccl``                                               |
+-----------+----------+--------------------------------------------------------+
| MIOpen    | miopen   | ``MIOpen``                                             |
+-----------+----------+--------------------------------------------------------+
| MIGraphX  | migraphx | ``migraphx::migraphx``, ``migraphx::migraphx_c``,      |
|           |          | ``migraphx::migraphx_cpu``, ``migraphx::migraphx_gpu``,|
|           |          | ``migraphx::migraphx_onnx``, ``migraphx::migraphx_tf`` |
+-----------+----------+--------------------------------------------------------+

Using CMake Presets
===================

CMake command-lines depending on how specific users like to be when compiling
code can grow to unwieldy lengths. This is the primary reason why projects tend
to bake script snippets into their build definitions controlling compiler
warning levels, changing CMake defaults (``CMAKE_BUILD_TYPE`` or
``BUILD_SHARED_LIBS`` just to name a few) and all sorts anti-patterns, all in
the name of convenience.

Load on the command-line interface (CLI) starts immediately by selecting a
toolchain, the set of utilities used to compile programs. To ease some of the
toolchain related pains, CMake does consult the ``CC`` and ``CXX`` environmental
variables when setting a default ``CMAKE_C[XX]_COMPILER`` respectively, but that
is just the tip of the iceberg. There's a fair number of variables related to
just the toolchain itself (typically supplied using
`toolchain files <https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html>`_
), and then we still haven't talked about user preference or project-specific
options.

IDEs supporting CMake (Visual Studio, Visual Studio Code, CLion, etc.) all came
up with their own way to register command-line fragments of different purpose in
a setup-and-forget fashion for quick assembly using graphical front-ends. This is
all nice, but configurations aren't portable, nor can they be reused in
Continuous Integration (CI) pipelines. CMake has condensed existing practice
into a portable JSON format that works in all IDEs and can be invoked from any
command-line. This is
`CMake Presets <https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html>`_
.

There are two types of preset files: one supplied by the project, called
``CMakePresets.json`` which is meant to be committed to version control,
typically used to drive CI; and one meant for the user to provide, called
``CMakeUserPresets.json``, typically used to house user preference and adapting
the build to the user's environment. These JSON files are allowed to include
other JSON files and the user presets always implicitly includes the non-user
variant.

Using HIP with presets
----------------------

Following is an example ``CMakeUserPresets.json`` file which actually compiles
the `amd/rocm-examples <https://github.com/amd/rocm-examples>`_ suite of sample
applications on a typical ROCm installation:

::

    {
      "version": 3,
      "cmakeMinimumRequired": {
        "major": 3,
        "minor": 21,
        "patch": 0
      },
      "configurePresets": [
        {
          "name": "layout",
          "hidden": true,
          "binaryDir": "${sourceDir}/build/${presetName}",
          "installDir": "${sourceDir}/install/${presetName}"
        },
        {
          "name": "generator-ninja-multi-config",
          "hidden": true,
          "generator": "Ninja Multi-Config"
        },
        {
          "name": "toolchain-makefiles-c/c++-amdclang",
          "hidden": true,
          "cacheVariables": {
            "CMAKE_C_COMPILER": "/opt/rocm/bin/amdclang",
            "CMAKE_CXX_COMPILER": "/opt/rocm/bin/amdclang++",
            "CMAKE_HIP_COMPILER": "/opt/rocm/bin/amdclang++"
          }
        },
        {
          "name": "clang-strict-iso-high-warn",
          "hidden": true,
          "cacheVariables": {
            "CMAKE_C_FLAGS": "-Wall -Wextra -pedantic",
            "CMAKE_CXX_FLAGS": "-Wall -Wextra -pedantic",
            "CMAKE_HIP_FLAGS": "-Wall -Wextra -pedantic"
          }
        },
        {
          "name": "ninja-mc-rocm",
          "displayName": "Ninja Multi-Config ROCm",
          "inherits": [
            "layout",
            "generator-ninja-multi-config",
            "toolchain-makefiles-c/c++-amdclang",
            "clang-strict-iso-high-warn"
          ]
        }
      ],
      "buildPresets": [
        {
          "name": "ninja-mc-rocm-debug",
          "displayName": "Debug",
          "configuration": "Debug",
          "configurePreset": "ninja-mc-rocm"
        },
        {
          "name": "ninja-mc-rocm-release",
          "displayName": "Release",
          "configuration": "Release",
          "configurePreset": "ninja-mc-rocm"
        },
        {
          "name": "ninja-mc-rocm-debug-verbose",
          "displayName": "Debug (verbose)",
          "configuration": "Debug",
          "configurePreset": "ninja-mc-rocm",
          "verbose": true
        },
        {
          "name": "ninja-mc-rocm-release-verbose",
          "displayName": "Release (verbose)",
          "configuration": "Release",
          "configurePreset": "ninja-mc-rocm",
          "verbose": true
        }
      ],
      "testPresets": [
        {
          "name": "ninja-mc-rocm-debug",
          "displayName": "Debug",
          "configuration": "Debug",
          "configurePreset": "ninja-mc-rocm",
          "execution": {
            "jobs": 0
          }
        },
        {
          "name": "ninja-mc-rocm-release",
          "displayName": "Release",
          "configuration": "Release",
          "configurePreset": "ninja-mc-rocm",
          "execution": {
            "jobs": 0
          }
        }
      ]
    }

.. note::
    Getting presets to work reliably on Windows requires some CMake improvements
    and/or support from compiler vendors. (Refer to 
    `Add support to the Visual Studio generators <https://gitlab.kitware.com/cmake/cmake/-/issues/24245>`_
    and `Sourcing environment scripts <https://gitlab.kitware.com/cmake/cmake/-/issues/21619>`_
    .)
