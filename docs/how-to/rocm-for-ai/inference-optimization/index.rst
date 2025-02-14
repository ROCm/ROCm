.. meta::
   :description: How to Use ROCm for AI inference optimization
   :keywords: ROCm, LLM, AI inference, Optimization, GPUs, usage, tutorial

*******************************************
Use ROCm for AI inference optimization
*******************************************

AI inference optimization is the process of improving the performance of machine learning models and speeding up the inference process. It includes:

- **Quantization**: This involves reducing the precision of model weights and activations while maintaining acceptable accuracy levels. Reduced precision improves inference efficiency because lower precision data requires less storage and better utilizes the hardware's computation power.  

- **Kernel optimization**: This technique involves optimizing computation kernels to exploit the underlying hardware capabilities. For example, the kernels can be optimized to use multiple GPU cores or utilize specialized hardware like tensor cores to accelerate the computations. 

- **Libraries**: Libraries such as Flash Attention, xFormers, and PyTorch TunableOp are used to accelerate deep learning models and improve the performance of inference workloads. 

- **Hardware acceleration**: Hardware acceleration techniques, like GPUs for AI inference, can significantly improve performance due to their parallel processing capabilities. 

- **Pruning**: This involves removing unnecessary connections, layers, or weights from a pre-trained model while maintaining acceptable accuracy levels, resulting in a smaller model that requires fewer computational resources to run inference. 

Utilizing these optimization techniques with the ROCm™ software platform can significantly reduce inference time, improve performance, and reduce the cost of your AI applications. 

Throughout the following topics, this guide discusses optimization techniques for inference workloads.

- :doc:`Model quantization <model-quantization>`

- :doc:`Model acceleration libraries <model-acceleration-libraries>`

- :doc:`Optimizing with Composable Kernel <optimizing-with-composable-kernel>`

- :doc:`Optimizing Triton kernels <optimizing-triton-kernel>`

- :doc:`Profiling and debugging <profiling-and-debugging>`

- :doc:`Workload tuning <workload>`

