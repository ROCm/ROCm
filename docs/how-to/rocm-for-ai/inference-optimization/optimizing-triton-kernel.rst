.. meta::
   :description: How to optimize Triton kernels for ROCm.
   :keywords: ROCm, LLM, fine-tuning, usage, MI300X, tutorial, Triton, kernel, performance, optimization

*************************
Optimizing Triton kernels
*************************

This section introduces the general steps for 
`Triton <https://openai.com/index/triton/>`_ kernel optimization. Broadly,
Triton kernel optimization is similar to :doc:`HIP <hip:how-to/performance_guidelines>`
and CUDA kernel optimization.

Refer to the
:ref:`Triton kernel performance optimization <mi300x-triton-kernel-performance-optimization>`
section of the :doc:`workload` guide
for detailed information.

Triton kernel performance optimization includes the following topics.

* :ref:`mi300x-autotunable-kernel-config`

* :ref:`mi300x-mlir-analysis`

* :ref:`mi300x-assembly-analysis`

* :ref:`mi300x-torchinductor-tuning`

* :ref:`mi300x-compute-kernel-occ`
