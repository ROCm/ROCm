.. meta::
   :description: Input-Output Memory Management Unit (IOMMU)
   :keywords: IOMMU, DMA, PCIe, xGMI, AMD, ROCm

****************************************************************
Input-Output Memory Management Unit (IOMMU)
****************************************************************

The I/O Memory Management Unit (IOMMU) provides memory remapping services for I/O devices. It adds support for address translation and system memory access protection on direct memory access (DMA) transfers from peripheral devices. 

The IOMMU's memory remapping services:

* provide private I/O space for devices used in a guest virtual machine.
* prevent unauthorized DMA requests to system memory and to memory-mapped I/O (MMIO).
* help in debugging memory access issues.
* facilitate peer-to-peer DMA.

The IOMMU also provides interrupt remapping, which is used by devices that support multiple interrupts and for interrupt delivery on hardware platforms with a large number of cores.

.. note::

  AMD Instinct accelerators are connected via XGMI links and don't use PCI/PCIe for peer-to-peer DMA. Because PCI/PCIe is not used for peer-to-peer DMA, there are no device physical addressing limitations or platform root port limitations. However, because non-GPU devices such as RDMA NICs use PCIe for peer-to-peer DMA, there might still be physical addressing and platform root port limitations when these non-GPU devices interact with other devices, including GPUs.

Linux supports IOMMU in both virtualized environments and bare metal. 

The IOMMU is enabled by default but can be disabled or put into passthrough mode through the Linux kernel command line:

.. list-table:: 
  :widths: 15 15 70
  :header-rows: 1

  * - IOMMU Mode
    - Kernel command
    - Description
  * - Enabled
    - Default setting
    - Recommended for AMD Radeon GPUs that need peer-to-peer DMA.
    
      The IOMMU is enabled in remapping mode. Each device gets its own I/O virtual address space. All devices on Linux register their DMA addressing capabilities, and the kernel will ensure that any address space mapped for DMA is mapped within the device's DMA addressing limits. Only address space explicitly mapped by the devices will be mapped into virtual address space. Attempts to access an unmapped page will generate an IOMMU page fault. 
  * - Passthrough
    - ``iommu=pt``
    - Recommended for AMD Instinct Accelerators and for AMD Radeon GPUs that don't need peer-to-peer DMA.

      Interrupt remapping is enabled but I/O remapping is disabled. The entire platform shares a common platform address space for system memory and MMIO spaces, ensuring compatibility with drivers from external vendors, while still supporting CPUs with a large number of cores. 
  * - Disabled
    - ``iommu=off``
    - Not recommended.
      
      The IOMMU is disabled and the entire platform shares a common platform address space for system memory and MMIO spaces.
      
      This mode should only be used with older Linux distributions with kernels that are not configured to support peer-to-peer DMA with an IOMMU. In these cases, the IOMMU needs to be disabled to use peer-to-peer DMA. 
    
The IOMMU also provides virtualized access to the MMIO portions of the platform address space for peer-to-peer DMA.

Because peer-to-peer DMA is not officially part of the PCI/PCIe specification, the behavior of peer-to-peer DMA varies between hardware platforms. 

AMD CPUs earlier than AMD Zen only supported peer-to-peer DMA for writes. On CPUs from AMD Zen and later, peer-to-peer DMA is fully supported. 

To use peer-to-peer DMA on Linux, enable the following options in your Linux kernel configuration:

* ``CONFIG_PCI_P2PDMA``
* ``CONFIG_DMABUF_MOVE_NOTIFY`` 
* ``CONFIG_HSA_AMD_P2P``
