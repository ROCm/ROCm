.. meta::
   :description: How ROCm uses PCIe atomics
   :keywords: PCIe, PCIe atomics, atomics, Atomic operations, AMD, ROCm

*****************************************************************************
How ROCm uses PCIe atomics
*****************************************************************************
AMD ROCm is an extension of the Heterogeneous System Architecture (HSA). To meet the requirements of an HSA-compliant system, ROCm supports queuing models, memory models, and signaling and synchronization protocols. ROCm can perform atomic Read-Modify-Write (RMW) transactions that extend inter-processor synchronization mechanisms to Input/Output (I/O) devices starting from Peripheral Component Interconnect Express 3.0 (PCIe™ 3.0). It supports the defined HSA capabilities for queuing and signaling memory operations. To learn more about the requirements of an HSA-compliant system, see the 
`HSA Platform System Architecture Specification <http://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf>`_.

ROCm uses platform atomics to perform memory operations like queuing, signaling, and synchronization across multiple CPU, GPU agents, and I/O devices. Platform atomics ensures that atomic operations run synchronously without interruptions or conflicts across multiple shared resources. Similarly, it enables memory alignment and synchronization of processing data within the shared memory locations and I/O devices.  This sections details how ROCm uses platform atomics and features of PCIe 3.0.

Platform atomics in ROCm
==============================
Platform atomics enable the set of atomic operations that perform RMW actions across multiple processors, devices, and memory locations so that they run synchronously without interruption. An atomic operation is a sequence of computing instructions run as a single, indivisible unit. These instructions are completed in their entirety without any interruptions. If the instructions can't be completed as a unit without interruption, none of the instructions are run. These operations support 32-bit and 64-bit address formats.

ROCm uses platform atomics to:

* Update the HSA queue's ``read_dispatch_id``. The command processor on the GPU agent uses a 64-bit atomic add operation. It updates the packet ID it processed.
* Update the HSA queue's ``write_dispatch_id``. The CPU and GPU agents use a 64-bit atomic add operation. It supports multi-writer queue insertions.
* Update HSA Signals. A 64-bit atomic operation is used for CPU & GPU synchronization.


PCIe for atomic operations
----------------------------
ROCm requires CPUs that support PCIe atomics. Similarly, all connected I/O devices should also support PCIe atomics for optimum compatibility. PCIe supports the ``CAS`` (Compare and Swap), ``FetchADD``, and ``SWAP`` atomic operations across multiple resources. These atomic operations are initiated by the I/O devices that support 32-bit, 64-bit, and 128-bit operands. Likewise, the target memory address where these atomic operations are performed should also be aligned to the size of the operand. This alignment ensures that the operations are performed efficiently and correctly without failure. 

When an atomic operation is successful, the requester receives a response of completion along with the operation result. However, any errors associated with the operation are signaled to the requester by updating the Completion Status field. Issues accessing the target location or executing the atomic operation are common errors. Depending upon the error, the Completion Status field is updated to Completer Abort (CA) or Unsupported Request (UR). The field is present in the Completion Descriptor.

To learn more about the industry standards and specifications of PCIe, see `PCI-SIG Specification <https://pcisig.com/specifications>`_.

To learn more about PCIe and its capabilities, consult the following white papers:

* `Atomic Read Modify Write Primitives by Intel <https://www.intel.es/content/dam/doc/white-paper/atomic-read-modify-write-primitives-i-o-devices-paper.pdf>`_
* `PCI Express 3 Accelerator White paper by Intel <https://www.intel.sg/content/dam/doc/white-paper/pci-express3-accelerator-white-paper.pdf>`_
* `PCIe Generation 4 Base Specification includes atomic operations <https://astralvx.com/storage/2020/11/PCI_Express_Base_4.0_Rev0.3_February19-2014.pdf>`_
* `Xilinx PCIe Ultrascale White paper <https://docs.xilinx.com/v/u/8OZSA2V1b1LLU2rRCDVGQw>`_

Working with PCIe 3.0 in ROCm
-------------------------------
Starting with PCIe 3.0, atomic operations can be requested, routed through, and completed by PCIe components. Routing and completion do not require software support. Component support for each can be identified by the Device Capabilities 2 (DevCap2) register. Upstream
bridges need to have atomic operations routing enabled. If not enabled, the atomic operations will fail even if the 
PCIe endpoint and PCIe I/O devices can perform atomic operations. 

If your system uses PCIe Express Switch to connect and enable communication between multiple PCIe components, the switch must also support atomic operations routing.

To enable atomic operations routing between multiple root ports, each root port must support atomic operation routing. This capability can be identified from the atomic operations routing supported bit in the DevCap2 register. If the bit has value of 1, routing is supported. Atomic operation requests are permitted only if a component's ``DEVCTL2.ATOMICOP_REQUESTER_ENABLE``
field is set. These requests can only be serviced if the upstream components also support atomic operation
completion, or can route it to a component that supports it, or both.  

ROCm uses the PCIe-ID-based ordering technology for peer-to-peer (P2P) data transmission. PCIe-ID-based ordering technology is used when the GPU initiates multiple write operations to different memory locations. For example. when there are two write operations which need to be run in this order::

1. Write to another GPU memory.
2. Write to system memory to indicate the transfer is complete.

The two write operations are routed to different locations. However, use of the ordering technology ensures that the order of the operation is maintained. 

For more information on changes implemented in PCIe 3.0, see `Overview of Changes to PCI Express 3.0 <https://www.mindshare.com/files/resources/PCIe%203-0.pdf>`_.





