===========================
How ROCm uses PCIe Atomics
===========================


ROCm PCIe Feature and Overview BAR Memory
==========================================


ROCm is an extension of HSA platform architecture, so it shares the queueing model, memory model, signaling and synchronization protocols. Platform atomics are integral to perform queuing and signaling memory operations where there may be multiple-writers across CPU and GPU agents.

The full list of HSA system architecture platform requirements are here: `HSA Sys Arch Features <http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf>`_.

The ROCm Platform uses the new PCI Express 3.0 (PCIe 3.0) features for Atomic Read-Modify-Write Transactions which extends inter-processor synchronization mechanisms to IO to support the defined set of HSA capabilities needed for queuing and signaling memory operations.

The new PCIe atomic operations operate as completers for ``CAS`` (Compare and Swap), ``FetchADD``, ``SWAP`` atomics. The atomic operations are initiated by the
I/O device which support 32-bit, 64-bit and 128-bit operand which target address have to be naturally aligned to operation sizes.

For ROCm the Platform atomics are used in ROCm in the following ways:

   * Update HSA queue’s read_dispatch_id: 64 bit atomic add used by the command processor on the GPU agent to update the packet ID it 	  processed.
   * Update HSA queue’s write_dispatch_id: 64 bit atomic add used by the CPU and GPU agent to support multi-writer queue insertions.
   * Update HSA Signals – 64bit atomic ops are used for CPU & GPU synchronization.

The PCIe 3.0 atomic operations feature allows atomic transactions to be requested by, routed through and completed by PCIe components. Routing and completion does not require software support. Component support for each is detectable via the DEVCAP2 register. Upstream bridges need to have atomic operations routing enabled or the Atomic Operations will fail even though PCIe endpoint and PCIe I/O Devices has the capability to Atomics Operations.

To do atomic operations routing capability between two or more Root Ports, each associated Root Port must indicate that capability via the atomic operations routing supported bit in the Device Capabilities 2 register.

If your system has a PCIe Express Switch it needs to support atomic operations routing. Atomic operations requests are permitted only if a component’s ``DEVCTL2.ATOMICOP_REQUESTER_ENABLE`` field is set. These requests can only be serviced if the upstream components support atomic operations completion and/or routing to a component which does. Atomic operations routing support=1, routing is supported; Atomic operations routing support=0, routing is not supported.

Atomic Operation is a Non-Posted transaction supporting 32-bit and 64-bit address formats, there must be a response for Completion containing the result of the operation. Errors associated with the operation (uncorrectable error accessing the target location or carrying out the Atomic operation) are signaled to the requester by setting the Completion Status field in the completion descriptor, they are set to to Completer Abort (CA) or Unsupported Request (UR).

To understand more about how PCIe Atomic operations work `PCIe Atomics <https://pcisig.com/sites/default/files/specification_documents/ECN_Atomic_Ops_080417.pdf>`_

`Linux Kernel Patch to pci_enable_atomic_request <https://patchwork.kernel.org/patch/7261731/>`_

There are also a number of papers which talk about these new capabilities:

  * `Atomic Read Modify Write Primitives by Intel <https://www.intel.es/content/dam/doc/white-paper/atomic-read-modify-write-primitives-i-o-devices-paper.pdf>`_
  * `PCI express 3 Accelerator Whitepaper by Intel <https://www.intel.sg/content/dam/doc/white-paper/pci-express3-accelerator-white-paper.pdf>`_
  * `Intel PCIe Generation 3 Hotchips Paper <https://www.hotchips.org/wp-content/uploads/hc_archives/hc21/1_sun/HC21.23.1.SystemInterconnectTutorial-Epub/HC21.23.131.Ajanovic-Intel-PCIeGen3.pdf>`_
  * `PCIe Generation 4 Base Specification includes Atomics Operation <https://astralvx.com/storage/2020/11/PCI_Express_Base_4.0_Rev0.3_February19-2014.pdf>`_

Other I/O devices with PCIe Atomics support

   * `Mellanox ConnectX-5 InfiniBand Card <http://www.mellanox.com/related-docs/prod_adapter_cards/PB_ConnectX-5_VPI_Card.pdf>`_
   * `Cray Aries Interconnect <http://www.hoti.org/hoti20/slides/Bob_Alverson.pdf>`_
   * `Xilinx PCIe Ultrascale Whitepaper <https://docs.xilinx.com/v/u/8OZSA2V1b1LLU2rRCDVGQw>`_
   * `Xilinx 7 Series Devices <https://docs.xilinx.com/v/u/1nfXeFNnGpA0ywyykvWHWQ>`_

Future bus technology with richer I/O Atomics Operation Support

  * `GenZ <http://genzconsortium.org/faq/gen-z-technology/#33/>`_

New PCIe Endpoints with support beyond AMD Ryzen and EPYC CPU; Intel Haswell or newer CPU’s with PCIe Generation 3.0 support.

  * `Mellanox Bluefield SOC <https://docs.nvidia.com/networking/display/BlueFieldSWv25111213/BlueField+Software+Overview>`_
  * `Cavium Thunder X2 <https://en.wikichip.org/wiki/cavium/thunderx2>`_

In ROCm, we also take advantage of PCIe ID based ordering technology for P2P when the GPU originates two writes to two different targets:  

  | 1. write to another GPU memory,
  
  | 2. then write to system memory to indicate transfer complete.

They are routed off to different ends of the computer but we want to make sure the write to system memory to indicate transfer complete occurs AFTER P2P write to GPU has complete.

`Good Paper on Understanding PCIe Generation 3 Throughput <https://www.altera.com/en_US/pdfs/literature/an/an690.pdf>`_

BAR Memory Overview
*******************
On a Xeon E5 based system in the BIOS we can turn on above 4GB PCIe addressing, if so he need to set MMIO Base address ( MMIOH Base) and Range ( MMIO High Size) in the BIOS.

In Supermicro system in the system bios you need to see the following

   * Advanced->PCIe/PCI/PnP configuration-> Above 4G Decoding = Enabled
  
   * Advanced->PCIe/PCI/PnP Configuration->MMIOH Base = 512G

   * Advanced->PCIe/PCI/PnP Configuration->MMIO High Size = 256G

When we support Large Bar Capability there is a Large Bar VBIOS which also disable the IO bar.

For GFX9 and Vega10 which have Physical Address up 44 bit and 48 bit Virtual address.

   * BAR0-1 registers: 64bit, prefetchable, GPU memory. 8GB or 16GB depending on Vega10 SKU. Must be placed < 2^44 to support P2P  	access from other Vega10.
   * BAR2-3 registers: 64bit, prefetchable, Doorbell. Must be placed < 2^44 to support P2P access from other Vega10.
   * BAR4 register: Optional, not a boot device.
   * BAR5 register: 32bit, non-prefetchable, MMIO. Must be placed < 4GB.

Here is how our BAR works on GFX 8 GPU’s with 40 bit Physical Address Limit ::

  11:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Fiji [Radeon R9 FURY / NANO Series] (rev c1)

  Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0b35
    
  Flags: bus master, fast devsel, latency 0, IRQ 119
    
  Memory at bf40000000 (64-bit, prefetchable) [size=256M]
   
  Memory at bf50000000 (64-bit, prefetchable) [size=2M]
   
  I/O ports at 3000 [size=256]
   
  Memory at c7400000 (32-bit, non-prefetchable) [size=256K]
   
  Expansion ROM at c7440000 [disabled] [size=128K]

Legend:

1 : GPU Frame Buffer BAR – In this example it happens to be 256M, but typically this will be size of the GPU memory (typically 4GB+). This BAR has to be placed < 2^40 to allow peer-to-peer access from other GFX8 AMD GPUs. For GFX9 (Vega GPU) the BAR has to be placed < 2^44 to allow peer-to-peer access from other GFX9 AMD GPUs.

2 : Doorbell BAR – The size of the BAR is typically will be < 10MB (currently fixed at 2MB) for this generation GPUs. This BAR has to be placed < 2^40 to allow peer-to-peer access from other current generation AMD GPUs.

3 : IO BAR - This is for legacy VGA and boot device support, but since this the GPUs in this project are not VGA devices (headless), this is not a concern even if the SBIOS does not setup.

4 : MMIO BAR – This is required for the AMD Driver SW to access the configuration registers. Since the reminder of the BAR available is only 1 DWORD (32bit), this is placed < 4GB. This is fixed at 256KB.

5 : Expansion ROM – This is required for the AMD Driver SW to access the GPU’s video-bios. This is currently fixed at 128KB.

For more information, you can review
`Overview of Changes to PCI Express 3.0 <https://www.mindshare.com/files/resources/PCIe%203-0.pdf>`_.
