---
myst:
  html_meta:
    "description": "Start building for HPC and AI with the performance-first AMD ROCm software stack. Explore how-to guides and reference docs."
    "keywords": "Radeon, open, compute, platform, install, how, conceptual, reference, home, docs"
---

# AMD ROCm documentation

ROCm is an open-source software platform optimized to extract HPC and AI workload
performance from AMD Instinct accelerators and AMD Radeon GPUs while maintaining
compatibility with industry software frameworks. For more information, see
[What is ROCm?](./what-is-rocm.rst)

ROCm supports multiple programming languages and programming interfaces such as
{doc}`HIP (Heterogeneous-Compute Interface for Portability)<hip:index>`, OpenCL,
and OpenMP, as explained in the [Programming guide](./how-to/programming_guide.rst).

If you're using AMD Radeon™ PRO or Radeon GPUs in a workstation setting with a display connected, review {doc}`Radeon-specific ROCm documentation<radeon:index>`.

ROCm documentation is organized into the following categories:

::::{grid} 1 2 2 2
:gutter: 3
:class-container: rocm-doc-grid

:::{grid-item-card} Install
:class-body: rocm-card-banner rocm-hue-2

* {doc}`ROCm on Linux <rocm-install-on-linux:reference/system-requirements>`
* {doc}`HIP SDK on Windows <rocm-install-on-windows:reference/system-requirements>`
* [ROCm on Radeon GPUs](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html)
* {doc}`Deep learning frameworks </how-to/deep-learning-rocm>`
* {doc}`Build from source </how-to/build-rocm>`
:::

:::{grid-item-card} How to
:class-body: rocm-card-banner rocm-hue-12

* [Use ROCm for AI](./how-to/rocm-for-ai/index.rst)
* [AI tutorials](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)
* [Use ROCm for HPC](./how-to/rocm-for-hpc/index.rst)
* [System optimization](./how-to/system-optimization/index.rst)
* [AMD Instinct MI300X performance validation and tuning](./how-to/tuning-guides/mi300x/index.rst)
* [System debugging](./how-to/system-debugging.md)
* [Use advanced compiler features](./conceptual/compiler-topics.md)
* [Set the number of CUs](./how-to/setting-cus)
* [Troubleshoot BAR access limitation](./how-to/Bar-Memory.rst)
* [ROCm examples](https://github.com/amd/rocm-examples)
:::

:::{grid-item-card} Conceptual
:class-body: rocm-card-banner rocm-hue-8

* [GPU architecture overview](./conceptual/gpu-arch.md)
* [Input-Output Memory Management Unit (IOMMU)](./conceptual/iommu.rst)
* [File structure (Linux FHS)](./conceptual/file-reorg.md)
* [GPU isolation techniques](./conceptual/gpu-isolation.md)
* [Using CMake](./conceptual/cmake-packages.rst)
* [PCIe atomics in ROCm](./conceptual/pcie-atomics.rst)
* [Inception v3 with PyTorch](./conceptual/ai-pytorch-inception.md)
* [Oversubscription of hardware resources](./conceptual/oversubscription.rst)
:::

:::{grid-item-card} Reference
:class-body: rocm-card-banner rocm-hue-6
<!-- markdownlint-disable MD051 -->
* [ROCm libraries](./reference/api-libraries.md)
* [ROCm tools, compilers, and runtimes](./reference/rocm-tools.md)
* [Environment variables](./reference/env-variables.rst)
* [Accelerator and  GPU hardware specifications](./reference/gpu-arch-specs.rst)
* [Precision support](./reference/precision-support.rst)
* [Graph safe support](./reference/graph-safe-support.rst)
<!-- markdownlint-enable MD051 -->
:::

::::
