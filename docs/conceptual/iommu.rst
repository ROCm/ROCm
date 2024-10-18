.. meta::
   :description: Input-Output Memory Management Unit (IOMMU)
   :keywords: IOMMU, DMA, PCIe, XGMI, AMD, ROCm

****************************************************************
Input-Output Memory Management Unit (IOMMU)
****************************************************************

Overview
================================================================

The IOMMU (IO Memory Management Unit) is a device which provides memory remapping services to 
IO devices on a platform. Similar to the MMU on a CPU, the IOMMU is able to provide a private 
virtual address space for each device or group of devices on a platform. While the IOMMU is 
most commonly used for virtualization provide a private IO space for devices used in a guest 
virtual machine, it can also be used to isolate devices on bare metal to prevent accidental
or nefarious DMA accesses to things that they should should not have access to. The IOMMU provides
virtualized access to the entire platform address space, not just system memory. It can also
provide virtualized access to the MMIO portions of the platform address space. This is used for
things like peer to peer DMA. In addition to DMA remapping, the IOMMU also support interrupt remapping.
In addition to virtualization use cases, this can be advantageous for devices which support multiple
interrupts on bare metal and it may be required for interrupt delivery on platforms with lots of cores.


Peer to Peer DMA
================================================================

Peer to Peer DMA over PCIe
----------------------------------------------------------------

Peer to Peer DMA is not officially part of the PCI/PCIe spec. As such, the behavior for peer to peer DMA
varies from platform to platform. AMD platforms prior to zen only supported peer to peer DMA for writes.
On Zen and newer CPUs, peer to peer DMA is fully supported. On Intel, peer to peer DMA is supported with
a number of limitations on various platforms. For example only certain root ports on the platform support it.
If one of your devices happens to be on an unsupported root port, it will not work. The Linux kernel has
special peer to peer DMA code to handle these cases. It provides an API for drivers to use which can look
at two devices and determine whether or not they can support peer to peer DMA. This functionality is controlled
by the ``CONFIG_PCI_P2PDMA`` kernel config option. In addition to that parameter, you also need to enable
``dma-buf`` (the framework for sharing DMA memory across drivers). To properly support peer to peer DMA with the IOMMU,
the following options must be enabled in your kernel config: ``CONFIG_PCI_P2PDMA``, ``CONFIG_DMABUF_MOVE_NOTIFY`` 
and ``CONFIG_HSA_AMD_P2P``.

Peer to Peer DMA over XGMI
----------------------------------------------------------------

Devices connected via XGMI links (MI50/60, MI100, MI200, MI300) have private XGMI links between GPUs so the
PCIe bus is not used for peer to peer DMA. As such, device physical addressing limitations and platform root
port limitations do not come into play. Note that non-GPU devices, such as RDMA NICs still use PCIe for peer
to peer DMA so the limitations mentioned in the section above would apply to interactions between the GPUs and those devices.

Linux IOMMU support
================================================================

Linux has supported the IOMMU for more than a decade in both virtualized environments and bare metal.
There are 3 modes of operation most commonly used:

* IOMMU disabled. The IOMMU is disabled and the entire platform shares a common platform address space for 
  system memory and MMIO spaces. Setting ``iommu=off`` on the kernel command line will disable the IOMMU.

* IOMMU enabled in passthrough mode. In this case, the IOMMU is enabled, but the identity mapping is used for all devices.
  The IOMMU is enabled, but the entire platform still shares a common platform address space for system memory and MMIO spaces.
  Setting ``iommu=pt`` on the kernel command line enables this mode.  There are cases where you many need some
  aspects of the IOMMU (e.g., interrupt remapping), but don't want DMA remapping or you have a system with a lot of cores and
  want good interrupt distribution across the cores.

* IOMMU enabled in remapping mode. In this mode, the IOMMU is enabled and each device gets it's own IO virtual address space.
  All devices on Linux register their DMA addressing capabilities (e.g. 32 bit, 40 bit, 44, bit, etc. i.e., how much address
  space can they natively access). The kernel will then make sure that any address space mapped for DMA is mapped within
  the DMA addressing limits of the device. Additionally, only address space explicitly mapped by the device is mapped into
  its virtual address space so an access to an unmapped page will generate an IOMMU page fault. This is generally the
  default when the IOMMU is enabled.

Advantages and disadvantages
================================================================

There advantages and disadvantages depending on the devices and platforms you are using.

Advantages
----------------------------------------------------------------

* Extra security and debuggability. Devices are prevented from access address space they are not allowed to access.
  This prevents accidental or nefarious device accesses to the platform address space.  Getting an IOMMU page fault on the
  bad access aids in debugging.

* Device DMA addressing limitations are no longer a problem. The kernel will guarantee that the device will always get
  DMA virtual addresses within the range of it's DMA addressing limits.  If you have a platform with a large amount of physical
  memory or if you have a platform where the MMIO aperture is above the DMA addressing limits of the device, some of
  the system memory may not be accessible or peer to peer DMA is not possible.  This is a common problem in multi-GPU scenarios.

Disadvantages
----------------------------------------------------------------

* There is added latency for device DMA to the platform address space due to the page table walks required to virtualize the IO address space.

* Peer to Peer DMA requires the kernel config options mentioned above to be enabled in the kernel config.
  Some older Linux distros do not enable all of those options so peer to peer DMA would not be available in that case.
  However, it should be noted that not using the IOMMU does not make peer to peer DMA just work.
  There may be platform limitations as noted above that you could run into unless you know the limitations of your platform and configure it appropriately.
