.. meta::
   :description: Learn about BAR configuration in AMD GPUs and ways to troubleshoot physical addressing limit
   :keywords: BAR memory, MMIO, GPU memory, Physical Addressing Limit, AMD, ROCm

**************************************
Troubleshoot BAR access limitation
**************************************
Direct Memory Access (DMA) to PCIe devices using Base Address Registers (BARs) can be restricted due to physical addressing limits. These restrictions can result in data access failures between the system components. Peer-to-peer (P2P) DMA is used to access resources such as registers and memory between devices. PCIe devices need memory-mapped input/output (MMIO) space for DMA, and these MMIO spaces are defined in the PCIe BARs.

These BARs are a set of 32-bit or 64-bit registers that are used to define the resources that PCIe devices provide. The CPU and other system devices also use these to access the resources of the PCIe devices. P2P DMA only works when one device can directly access the local BAR memory of another. If the memory address of a BAR memory exceeds the physical addressing limit of a device, the device will not be able to access that BAR. This could be the device's own BAR or the BAR of another device in the system.

If the BAR memory exceeds than the physical addressing limit of the device, the device will not be able to access the remote BAR. 

To handle any BAR access issues that might occur, you need to be aware of the physical address limitations of the devices and understand the :ref:`BAR configuration of AMD GPUs <bar-configuration>`. This information is important when setting up additional MMIO apertures for PCIe devices in the system's physical address space.

Handling physical address limitation
=============================================
When a system boots, the system BIOS allocates the physical address space for the components in the system, including system memory and MMIO apertures. On modern 64-bit platforms, there are generally two or more MMIO apertures: one located below 4 GB of physical address space for 32-bit compatibility, and one or more above 4 GB for devices needing more space.

You can control the memory address of the high MMIO aperture from the system BIOS configuration options. This lets you configure the additional MMIO space to align with the physical addressing limit and allows P2P DMA between the devices. For example, if a PCIe device is limited to 44-bit of physical addressing, you should ensure that the MMIO aperture is set below 44-bit in the system physical address space.

There are two ways to handle this:

* Ensure that the high MMIO aperture is within the physical addressing limits of the devices in the system. For example, if the devices have a 44-bit physical addressing limit, set the ``MMIO High Base`` and ``MMIO High size`` options in the BIOS such that the aperture is within the 44-bit address range, and ensure that the ``Above 4G Decoding`` option is Enabled.  

* Enable the Input-Output Memory Management Unit (IOMMU). When the IOMMU is enabled in non-passthrough mode, it will create a virtual I/O address space for each device on the system. It also ensures that all virtual addresses created in that space are within the physical addressing limits of the device. For more information on IOMMU, see :doc:`../conceptual/iommu`. 

.. _bar-configuration:

BAR configuration for AMD GPUs
================================================

The following table shows how the BARs are configured for AMD GPUs.


.. list-table:: 
  :widths: 25 25 50
  :header-rows: 1

  * - BAR Type
    - Value
    - Description
  * - BAR0-1 registers
    - 64-bit, Prefetchable, GPU memory
    - 8 GB or 16 GB depending on GPU. Set to less than 2^44 to support P2P access from other GPUs with a 44-bit physical address limit. Prefetchable memory enables faster read operation for high-performance computing (HPC) by fetching the contiguous data from the same data source even before requested as an anticipation of a future request.
  * - BAR2-3 registers
    - 64-bit, Prefetchable, Doorbell
    - Set to less than 2^44 to support P2P access from other GPUs with a 44-bit physical address limit. As a Doorbell BAR, it indicates to the GPU that a new operation is in its queue to be processed. 
  * - BAR4 register
    - Optional
    - Not a boot device
  * - BAR5 register
    - 32-bit, Non-prefetchable, MMIO
    - Is set to less than 4 GB.

Example of BAR usage on AMD GPUs
-------------------------------------
Following is an example configuration of BARs set by the system BIOS on GFX8 GPUs with the 40-bit physical addressing limit: 

.. code:: shell 

  11:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Fiji [Radeon R9 FURY / NANO
  Series] (rev c1)

  Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0b35

  Flags: bus master, fast devsel, latency 0, IRQ 119

  Memory at bf40000000 (64-bit, prefetchable) [size=256M]

  Memory at bf50000000 (64-bit, prefetchable) [size=2M]

  I/O ports at 3000 [size=256]

  Memory at c7400000 (32-bit, non-prefetchable) [size=256K]

  Expansion ROM at c7440000 [disabled] [size=128K]

Details of the BARs configured in the example are: 

**GPU Frame Buffer BAR:** ``Memory at bf40000000 (64-bit, prefetchable) [size=256M]``

The size of the BAR in the example is 256 MB. Generally, it will be the size of the GPU memory (typically 4 GB+). Depending upon the physical address limit and generation of AMD GPUs, the BAR can be set below 2^40, 2^44, or 2^48. 

**Doorbell BAR:** ``Memory at bf50000000 (64-bit, prefetchable) [size=2M]``

The size of the BAR should typically be less than 10 MB for this generation of GPUs and has been set to 2 MB in the example. This BAR is placed less than 2^40 to allow peer-to-peer access from other generations of AMD GPUs.

**I/O BAR:** ``I/O ports at 3000 [size=256]``

This is for legacy VGA and boot device support. Because the GPUs used are not connected to a display (VGA devices), this is not a concern, even if it isn't set up in the system BIOS.

**MMIO BAR:** ``Memory at c7400000 (32-bit, non-prefetchable) [size=256K]``

The AMD Driver requires this to access the configuration registers. Since the reminder of the BAR available is only 1 DWORD (32-bit), this is set less than 4 GB. In the example, it is fixed at 256 KB.

**Expansion ROM:** ``Expansion ROM at c7440000 [disabled] [size=128K]``

This is required by the AMD Driver to access the GPU video-BIOS. In the example, it is fixed at 128 KB.