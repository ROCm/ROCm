.. meta::
   :description: How ROCm uses PCIe atomics
   :keywords: PCIe, PCIe atomics, atomics, Atomic operations, AMD, ROCm

*****************************************************************************
How ROCm uses PCIe atomics
*****************************************************************************
AMD ROCm is an extension of the Heterogeneous System Architecture (HSA). To meet the requirements of an HSA-compliant system, ROCm supports queuing models, memory models, and signaling and synchronization protocols. ROCm can perform atomic Read-Modify-Write (RMW) transactions that extend inter-processor synchronization mechanisms to I/O devices starting from Peripheral Component Interconnect Express 3.0 (PCIe™ 3.0). It supports the defined HSA capabilities for queuing and signaling memory operations. To learn more about the requirements of an HSA-compliant system, see the 
`HSA Platform System Architecture Specification <http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf>`_.

ROCm uses platform atomics to perform memory operations like queuing, signaling, and synchronization across multiple CPU, GPU agents, and I/O devices. Platform atomics ensures that atomic operations run synchronously without interruptions or conflicts across multiple shared resources. Similarly, it enables memory alignment and synchronization of processing data within the shared memory locations and Input/Output (I/O) devices.  This sections details how ROCm uses platform atomics and features of PCIe 3.0.

Platform atomics in ROCm
==============================
Platform atomics enables the set of atomic operations that perform RMW actions across multiple processors, devices, and memory locations to execute synchronously without interruption. An atomic operation is a sequence of computing instructions executed as a single, indivisible unit. These instructions are completed in their entirety without any interruptions or are not executed. These operations support 32-bit and 64-bit address formats.

ROCm uses platform atomics:

* Update the HSA queue's ``read_dispatch_id``. The command processor on the GPU agent uses a 64-bit atomic add operation. It updates the packet ID it processed.
* Update the HSA queue's ``write_dispatch_id``. The CPU and GPU agents use a 64-bit atomic add operation. It supports multi-writer queue insertions.
* Update HSA Signals. A 64-bit atomic operation is used for CPU & GPU synchronization.


PCIe for atomic operations
----------------------------

PCIe supports the ``CAS`` (Compare and Swap), ``FetchADD``, and ``SWAP`` atomic operations across multiple resource. These atomic operations are initiated by the I/O devices that support 32-bit, 64-bit, and 128-bit operands. Likewise, the target memory address where these atomic operations are performed should also be aligned to the size of the operand. This alignment ensures that the operations are performed efficiently and correctly without failure. 

When an atomic operation is successful, the requester receives a response of completion along with the operation result. However, any errors associated with the operation, are signaled to the requester by updating the Completion Status field. Some common errors can be issues accessing the target location or executing the atomic operation. Depending upon the error, the Completion Status field is updated to Completer Abort (CA) or Unsupported Request (UR). The field is present in the Completion Descriptor.

To learn more about the industry standards and specifications of PCIe, see `PCI-SIG Specification <https://pcisig.com/specifications>`_.

To learn more about PCIe and its capabilities, consult the following white papers:

* `Atomic Read Modify Write Primitives by Intel <https://www.intel.es/content/dam/doc/white-paper/atomic-read-modify-write-primitives-i-o-devices-paper.pdf>`_
* `PCI Express 3 Accelerator White paper by Intel <https://www.intel.sg/content/dam/doc/white-paper/pci-express3-accelerator-white-paper.pdf>`_
* `PCIe Generation 4 Base Specification includes atomic operations <https://astralvx.com/storage/2020/11/PCI_Express_Base_4.0_Rev0.3_February19-2014.pdf>`_
* `Xilinx PCIe Ultrascale White paper <https://docs.xilinx.com/v/u/8OZSA2V1b1LLU2rRCDVGQw>`_

Working of PCIe 3.0 in ROCm
-------------------------------
Starting from PCIe 3.0, atomic operations can be requested, routed through, and completed by PCIe components. Routing and completion do not require software support. Component support for each can be identified by the Device Capabilities 2 (DevCap2) register. Upstream
bridges need to have atomic operations routing enabled. If not enabled, the atomic operations will fail even if the 
PCIe endpoint and PCIe I/O devices can perform atomic operations.

To enable atomic operations routing between multiple Root Ports, each Root Port must support atomic operations routing. This capability can be identified from the atomic operations routing supported bit in the DevCap2 register. If the bit has value of 1, routing is supported.

If your system has a PCIe Express Switch, it must support atomic operations routing. Atomic
operations requests are permitted only if a component's ``DEVCTL2.ATOMICOP_REQUESTER_ENABLE``
field is set. These requests can only be serviced if the upstream components also support atomic operation
completion, can route it to a component that supports it, or both. 

ROCm also uses the PCIe-ID-based ordering technology for peer-to-peer (P2P) data transmission when the GPU
initiates two write operations to two memory locations. As an example scenario, there are two write operations which needs to be executed in the defined order:

1. Write to another GPU memory.
2. Write to system memory, to indicate the transfer is complete.

The two write operations are routed to different locations. However, use of the ordering technology ensures that the order of the operation is maintained. 

For more information on changes implemented in PCIe 3.0, see `Overview of Changes to PCI Express 3.0 <https://www.mindshare.com/files/resources/PCIe%203-0.pdf>`_.

CPUs and I/O devices with PCIe atomics support
------------------------------------------------
ROCm requires CPUs that support PCIe atomics. Modern CPUs after the release of first generation AMD Zen CPU and Intel™ Haswell support PCIe atomics. Some of the PCIe Endpoints with support beyond AMD Ryzen, AMD EPYC, Intel™ Haswell, or newer CPUs with PCIe 3.0 support are:

* Mellanox Bluefield SOC
* Cavium Thunder X2

Along with the CPUs, all the connected I/O devices should also support PCIe atomics for optimum compatibility and execution of atomic operations. Some of the I/O devices with PCIe atomic support are: 

* Mellanox ConnectX-5 InfiniBand Card
* Cray Aries Interconnect
* Xilinx 7 Series






