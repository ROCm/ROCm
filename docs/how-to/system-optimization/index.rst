.. meta::
   :description: Learn about AMD hardware optimization for HPC-specific and workstation workloads.
   :keywords: high-performance computing, HPC, Instinct accelerators, Radeon,
              tuning, tuning guide, AMD, ROCm

*******************
System optimization
*******************

This guide outlines system setup and tuning suggestions for AMD hardware to
optimize performance for specific types of workloads or use-cases.

High-performance computing workloads
====================================

High-performance computing (HPC) workloads have unique requirements. The default
hardware and BIOS configurations for OEM platforms may not provide optimal
performance for HPC workloads. To enable optimal HPC settings on a per-platform
and per-workload level, this chapter describes:

* BIOS settings that can impact performance
* Hardware configuration best practices
* Supported versions of operating systems
* Workload-specific recommendations for optimal BIOS and operating system
  settings

There is also a discussion on the AMD Instinct™ software development
environment, including information on how to install and run the DGEMM, STREAM,
HPCG, and HPL benchmarks. This guide provides a good starting point but is
not tested exhaustively across all compilers.

Knowledge prerequisites to better understand this document and to perform tuning
for HPC applications include:

* Experience in configuring servers
* Administrative access to the server's Management Interface (BMC)
* Administrative access to the operating system
* Familiarity with the OEM server's BMC (strongly recommended)
* Familiarity with the OS specific tools for configuration, monitoring, and
  troubleshooting (strongly recommended)

This document provides guidance on tuning systems with various AMD Instinct
accelerators for HPC workloads. The following sections don't comprise an
all-inclusive guide, and some items referred to may have similar, but different,
names in various OEM systems (for example, OEM-specific BIOS settings). This
following sections also provide suggestions on items that should be the initial
focus of additional, application-specific tuning.

While this guide is a good starting point, developers are encouraged to perform
their own performance testing for additional tuning.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - System optimization guide

     - Architecture reference

     - White papers

   * - :doc:`AMD Instinct MI300X <mi300x>`

     - `AMD Instinct MI300 instruction set architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_

     - `CDNA 3 architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>`_

   * - :doc:`AMD Instinct MI300A <mi300a>`

     - `AMD Instinct MI300 instruction set architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_

     - `CDNA 3 architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>`_

   * - :doc:`AMD Instinct MI200 <mi200>`

     - `AMD Instinct MI200 instruction set architecture <https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf>`_

     - `CDNA 2 architecture <https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf>`_

   * - :doc:`AMD Instinct MI100 <mi100>`

     - `AMD Instinct MI100 instruction set architecture <https://www.amd.com/system/files/TechDocs/instinct-mi100-cdna1-shader-instruction-set-architecture%C2%A0.pdf>`_

     - `CDNA architecture <https://www.amd.com/system/files/documents/amd-cdna-whitepaper.pdf>`_

Workstation workloads
=====================

Workstation workloads, much like those for HPC, have a unique set of
requirements: a blend of both graphics and compute, certification, stability and
others.

The document covers specific software requirements and processes needed to use
these GPUs for Single Root I/O Virtualization (SR-IOV) and machine learning
tasks.

The main purpose of this document is to help users utilize the RDNA™ 2 GPUs to
their full potential.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - System optimization guide

     - Architecture reference

     - White papers

   * - :doc:`AMD Radeon PRO W6000 and V620 <w6000-v620>`

     - `AMD RDNA 2 instruction set architecture <https://www.amd.com/system/files/TechDocs/rdna2-shader-instruction-set-architecture.pdf>`_

     - `RDNA 2 architecture <https://www.amd.com/system/files/documents/rdna2-explained-radeon-pro-W6000.pdf>`_

