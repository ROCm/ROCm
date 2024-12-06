<head>
  <meta charset="UTF-8">
  <meta name="description" content="AMD ROCm documentation">
  <meta name="keywords" content="documentation, guides, installation, compatibility, support,
  reference, ROCm, AMD">
</head>

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

* [Programming guide](./how-to/hip_programming_guide.rst)
* [Use ROCm for AI](./how-to/rocm-for-ai/index.rst)
* [Use ROCm for HPC](./how-to/rocm-for-hpc/index.rst)
* [Fine-tune LLMs and inference optimization](./how-to/llm-fine-tuning-optimization/index.rst)
* [System optimization](./how-to/system-optimization/index.rst)
* [AMD Instinct MI300X performance validation and tuning](./how-to/tuning-guides/mi300x/index.rst)
* [GPU cluster networking](https://rocm.docs.amd.com/projects/gpu-cluster-networking/en/latest/index.html)
* [System debugging](./how-to/system-debugging.md)
* [Use MPI](./how-to/gpu-enabled-mpi.rst)
* [Use advanced compiler features](./conceptual/compiler-topics.md)
* [Set the number of CUs](./how-to/setting-cus)  
* [ROCm examples](https://github.com/amd/rocm-examples)
:::

:::{grid-item-card} Conceptual
:class-body: rocm-card-banner rocm-hue-8

* [GPU architecture overview](./conceptual/gpu-arch.md)
* [Input-Output Memory Management Unit (IOMMU)](./conceptual/iommu.rst)
* [File structure (Linux FHS)](./conceptual/file-reorg.md)
* [GPU isolation techniques](./conceptual/gpu-isolation.md)
* [Using CMake](./conceptual/cmake-packages.rst)
* [ROCm & PCIe atomics](./conceptual/More-about-how-ROCm-uses-PCIe-Atomics.rst)
* [Inception v3 with PyTorch](./conceptual/ai-pytorch-inception.md)
* [Oversubscription of hardware resources](./conceptual/oversubscription.rst)
:::

<!-- markdownlint-disable MD051 -->
:::{grid-item-card} Reference
:class-body: rocm-card-banner rocm-hue-6

* [ROCm libraries](./reference/api-libraries.md)
* [ROCm tools, compilers, and runtimes](./reference/rocm-tools.md)
* [Accelerator and  GPU hardware specifications](./reference/gpu-arch-specs.rst)
* [Precision support](./reference/precision-support.rst)
:::
<!-- markdownlint-enable MD051 -->

::::
