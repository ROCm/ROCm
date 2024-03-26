# ROCm Compilers Disambiguation

ROCm ships multiple compilers of varying origins and purposes. This article
disambiguates compiler naming used throughout the documentation.

## Compiler Terms

| Term | Description |
| - | - |
| `amdclang++` | Clang/LLVM-based compiler that is part of `rocm-llvm` package. The source code is available at <a href="https://github.com/RadeonOpenCompute/llvm-project" target="_blank">https://github.com/RadeonOpenCompute/llvm-project</a>. |
| AOCC | Closed-source clang-based compiler that includes additional CPU optimizations. Offered as part of ROCm via the `rocm-llvm-alt` package. See for details, <a href="https://developer.amd.com/amd-aocc/" target="_blank">https://developer.amd.com/amd-aocc/</a>. |
| HIP-Clang | Informal term for the `amdclang++` compiler |
| HIPIFY | Tools including `hipify-clang` and `hipify-perl`, used to automatically translate CUDA source code into portable HIP C++. The source code is available at <a href="https://github.com/ROCm-Developer-Tools/HIPIFY" target="_blank">https://github.com/ROCm-Developer-Tools/HIPIFY</a> |
| `hipcc` | HIP compiler driver. A utility that invokes `clang` or `nvcc` depending on the target and passes the appropriate include and library options for the target compiler and HIP infrastructure. The source code is available at <a href="https://github.com/ROCm-Developer-Tools/HIPCC" target="_blank">https://github.com/ROCm-Developer-Tools/HIPCC</a>. |
| ROCmCC | Clang/LLVM-based compiler. ROCmCC in itself is not a binary but refers to the overall compiler. |
