<head>
  <meta charset="UTF-8">
  <meta name="description" content="AMD ROCm documentation">
  <meta name="keywords" content="documentation, guides, installation, compatibility, support,
  reference, ROCm, AMD">
</head>

# AMD ROCm documentation

ROCm is an open-source software platform optimized to extract HPC and AI workload
performance from AMD Instinct accelerators and AMD Radeon GPUs while maintaining
compatibility with industry software frameworks. For more information, see [What is ROCm?](./what-is-rocm.rst)

If you're using Radeon GPUs, consider reviewing {doc}`Radeon-specific ROCm documentation<radeon:index>`.

Installation instructions are available from:

* {doc}`ROCm installation for Linux<rocm-install-on-linux:index>`
* {doc}`HIP SDK installation for Windows<rocm-install-on-windows:index>`
* [Deep learning frameworks installation](./how-to/deep-learning-rocm.rst)
* [Build ROCm from source](./how-to/build-rocm.rst)

ROCm documentation is organized into the following categories:

::::{grid} 1 2 2 2
:gutter: 3
:class-container: rocm-doc-grid

:::{grid-item-card} Compatibility
:class-body: rocm-card-banner rocm-hue-2

* [Compatibility matrix](./compatibility/compatibility-matrix.rst)
* {doc}`Linux system requirements<rocm-install-on-linux:reference/system-requirements>`
* {doc}`Windows system requirements<rocm-install-on-windows:reference/system-requirements>`
* {doc}`Third-party support<rocm-install-on-linux:reference/3rd-party-support-matrix>`
* {doc}`User and kernel-space support matrix<rocm-install-on-linux:reference/user-kernel-space-compat-matrix>`
* {doc}`Docker image support matrix<rocm-install-on-linux:reference/docker-image-support-matrix>`
* {doc}`Use ROCm on Radeon GPUs<radeon:index>`
:::

:::{grid-item-card} How to
:class-body: rocm-card-banner rocm-hue-12

* [Using ROCm for AI](./how-to/rocm-for-ai/index.rst)
* [Using ROCm for HPC](./how-to/rocm-for-hpc/index.rst)
* [Fine-tuning LLMs and inference optimization](./how-to/llm-fine-tuning-optimization/index.rst)
* [System optimization](./how-to/system-optimization/index.rst)
* [AMD Instinct MI300X performance validation and tuning](./how-to/tuning-guides/mi300x/index.rst)
* [GPU cluster networking](https://rocm.docs.amd.com/projects/gpu-cluster-networking/en/latest/index.html)
* [System debugging](./how-to/system-debugging.md)
* [Using MPI](./how-to/gpu-enabled-mpi.rst)
* [Using advanced compiler features](./conceptual/compiler-topics.md)
* [Setting the number of CUs](./how-to/setting-cus)  
* [ROCm examples](https://github.com/amd/rocm-examples)
:::

:::{grid-item-card} Conceptual
:class-body: rocm-card-banner rocm-hue-8

* [GPU architecture overview](./conceptual/gpu-arch.md)
* [GPU memory](./conceptual/gpu-memory.md)
* [File structure (Linux FHS)](./conceptual/file-reorg.md)
* [GPU isolation techniques](./conceptual/gpu-isolation.md)
* [Using CMake](./conceptual/cmake-packages.rst)
* [PCIe atomics in ROCm](./conceptual/pcie-atomics.rst)
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
