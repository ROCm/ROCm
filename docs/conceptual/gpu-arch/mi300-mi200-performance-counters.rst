.. meta::
  :description: MI300 and MI200 series performance counters and metrics
  :keywords: MI300, MI200, performance counters, command processor counters

***************************************************************************************************
MI300 and MI200 series performance counters and metrics
***************************************************************************************************

This document lists and describes the hardware performance counters and derived metrics available
for the AMD Instinct™ MI300 and MI200 GPU. You can also access this information using the
:doc:`ROCProfiler tool <rocprofiler:rocprofv1>`.

MI300 and MI200 series performance counters
===============================================================

Series performance counters include the following categories:

* :ref:`command-processor-counters`
* :ref:`graphics-register-bus-manager-counters`
* :ref:`spi-counters`
* :ref:`compute-unit-counters`
* :ref:`l1i-and-sl1d-cache-counters`
* :ref:`vector-l1-cache-subsystem-counters`
* :ref:`l2-cache-access-counters`

The following sections provide additional details for each category.

.. note::

  Preliminary validation of all MI300 and MI200 series performance counters is in progress. Those with
  an asterisk (*) require further evaluation.

.. _command-processor-counters:

Command processor counters
---------------------------------------------------------------------------------------------------------------

Command processor counters are further classified into command processor-fetcher and command
processor-compute.

Command processor-fetcher counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``CPF_CMP_UTCL1_STALL_ON_TRANSLATION``", "Cycles", "Number of cycles one of the compute unified translation caches (L1) is stalled waiting on translation"
  "``CPF_CPF_STAT_BUSY``", "Cycles", "Number of cycles command processor-fetcher is busy"
  "``CPF_CPF_STAT_IDLE``", "Cycles", "Number of cycles command processor-fetcher is idle"
  "``CPF_CPF_STAT_STALL``", "Cycles", "Number of cycles command processor-fetcher is stalled"
  "``CPF_CPF_TCIU_BUSY``", "Cycles", "Number of cycles command processor-fetcher texture cache interface unit interface is busy"
  "``CPF_CPF_TCIU_IDLE``", "Cycles", "Number of cycles command processor-fetcher texture cache interface unit interface is idle"
  "``CPF_CPF_TCIU_STALL``", "Cycles", "Number of cycles command processor-fetcher texture cache interface unit interface is stalled waiting on free tags"

The texture cache interface unit is the interface between the command processor and the memory
system.

Command processor-compute counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``CPC_ME1_BUSY_FOR_PACKET_DECODE``", "Cycles", "Number of cycles command processor-compute micro engine is busy decoding packets"
  "``CPC_UTCL1_STALL_ON_TRANSLATION``", "Cycles", "Number of cycles one of the unified translation caches (L1) is stalled waiting on translation"
  "``CPC_CPC_STAT_BUSY``", "Cycles", "Number of cycles command processor-compute is busy"
  "``CPC_CPC_STAT_IDLE``", "Cycles", "Number of cycles command processor-compute is idle"
  "``CPC_CPC_STAT_STALL``", "Cycles", "Number of cycles command processor-compute is stalled"
  "``CPC_CPC_TCIU_BUSY``", "Cycles", "Number of cycles command processor-compute texture cache interface unit interface is busy"
  "``CPC_CPC_TCIU_IDLE``", "Cycles", "Number of cycles command processor-compute texture cache interface unit interface is idle"
  "``CPC_CPC_UTCL2IU_BUSY``", "Cycles", "Number of cycles command processor-compute unified translation cache (L2) interface is busy"
  "``CPC_CPC_UTCL2IU_IDLE``", "Cycles", "Number of cycles command processor-compute unified translation cache (L2) interface is idle"
  "``CPC_CPC_UTCL2IU_STALL``", "Cycles", "Number of cycles command processor-compute unified translation cache (L2) interface is stalled"
  "``CPC_ME1_DC0_SPI_BUSY``", "Cycles", "Number of cycles command processor-compute micro engine processor is busy"

The micro engine runs packet-processing firmware on the command processor-compute counter.

.. _graphics-register-bus-manager-counters:

Graphics register bus manager counters
---------------------------------------------------------------------------------------------------------------

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``GRBM_COUNT``", "Cycles","Number of free-running GPU cycles"
  "``GRBM_GUI_ACTIVE``", "Cycles", "Number of GPU active cycles"
  "``GRBM_CP_BUSY``", "Cycles", "Number of cycles any of the command processor blocks are busy"
  "``GRBM_SPI_BUSY``", "Cycles", "Number of cycles any of the shader processor input is busy in the shader engines"
  "``GRBM_TA_BUSY``", "Cycles", "Number of cycles any of the texture addressing unit is busy in the shader engines"
  "``GRBM_TC_BUSY``", "Cycles", "Number of cycles any of the texture cache blocks are busy"
  "``GRBM_CPC_BUSY``", "Cycles", "Number of cycles the command processor-compute is busy"
  "``GRBM_CPF_BUSY``", "Cycles", "Number of cycles the command processor-fetcher is busy"
  "``GRBM_UTCL2_BUSY``", "Cycles", "Number of cycles the unified translation cache (Level 2 [L2]) block is busy"
  "``GRBM_EA_BUSY``", "Cycles", "Number of cycles the efficiency arbiter block is busy"

Texture cache blocks include:

* Texture cache arbiter
* Texture cache per pipe, also known as vector Level 1 (L1) cache
* Texture cache per channel, also known as known as L2 cache
* Texture cache interface

.. _spi-counters:

Shader processor input counters
---------------------------------------------------------------------------------------------------------------

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SPI_CSN_BUSY``", "Cycles", "Number of cycles with outstanding waves"
  "``SPI_CSN_WINDOW_VALID``", "Cycles", "Number of cycles enabled by ``perfcounter_start`` event"
  "``SPI_CSN_NUM_THREADGROUPS``", "Workgroups", "Number of dispatched workgroups"
  "``SPI_CSN_WAVE``", "Wavefronts", "Number of dispatched wavefronts"
  "``SPI_RA_REQ_NO_ALLOC``", "Cycles", "Number of arbiter cycles with requests but no allocation"
  "``SPI_RA_REQ_NO_ALLOC_CSN``", "Cycles", "Number of arbiter cycles with compute shader (n\ :sup:`th` pipe) requests but no compute shader (n\ :sup:`th` pipe) allocation"
  "``SPI_RA_RES_STALL_CSN``", "Cycles", "Number of arbiter stall cycles due to shortage of compute shader (n\ :sup:`th` pipe) pipeline slots"
  "``SPI_RA_TMP_STALL_CSN``", "Cycles", "Number of stall cycles due to shortage of temp space"
  "``SPI_RA_WAVE_SIMD_FULL_CSN``", "SIMD-cycles", "Accumulated number of single instruction, multiple data (SIMD) per cycle affected by shortage of wave slots for compute shader (n\ :sup:`th` pipe) wave dispatch"
  "``SPI_RA_VGPR_SIMD_FULL_CSN``", "SIMD-cycles", "Accumulated number of SIMDs per cycle affected by shortage of vector general-purpose register (VGPR) slots for compute shader (n\ :sup:`th` pipe) wave dispatch"
  "``SPI_RA_SGPR_SIMD_FULL_CSN``", "SIMD-cycles", "Accumulated number of SIMDs per cycle affected by shortage of scalar general-purpose register (SGPR) slots for compute shader (n\ :sup:`th` pipe) wave dispatch"
  "``SPI_RA_LDS_CU_FULL_CSN``", "CU", "Number of compute units affected by shortage of local data share (LDS) space for compute shader (n\ :sup:`th` pipe) wave dispatch"
  "``SPI_RA_BAR_CU_FULL_CSN``", "CU", "Number of compute units with compute shader (n\ :sup:`th` pipe) waves waiting at a BARRIER"
  "``SPI_RA_BULKY_CU_FULL_CSN``", "CU", "Number of compute units with compute shader (n\ :sup:`th` pipe) waves waiting for BULKY resource"
  "``SPI_RA_TGLIM_CU_FULL_CSN``", "Cycles", "Number of compute shader (n\ :sup:`th` pipe) wave stall cycles due to restriction of ``tg_limit`` for thread group size"
  "``SPI_RA_WVLIM_STALL_CSN``", "Cycles", "Number of cycles compute shader (n\ :sup:`th` pipe) is stalled due to ``WAVE_LIMIT``"
  "``SPI_VWC_CSC_WR``", "Qcycles", "Number of quad-cycles taken to initialize VGPRs when launching waves"
  "``SPI_SWC_CSC_WR``", "Qcycles", "Number of quad-cycles taken to initialize SGPRs when launching waves"

.. _compute-unit-counters:

Compute unit counters
---------------------------------------------------------------------------------------------------------------

The compute unit counters are further classified into instruction mix, matrix fused multiply-add (FMA)
operation counters, level counters, wavefront counters, wavefront cycle counters, and LDS counters.

Instruction mix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_INSTS``", "Instr", "Number of instructions issued"
  "``SQ_INSTS_VALU``", "Instr", "Number of vector arithmetic logic unit (VALU) instructions including matrix FMA issued"
  "``SQ_INSTS_VALU_ADD_F16``", "Instr", "Number of VALU half-precision floating-point (F16) ``ADD`` or ``SUB`` instructions issued"
  "``SQ_INSTS_VALU_MUL_F16``", "Instr", "Number of VALU F16 Multiply instructions issued"
  "``SQ_INSTS_VALU_FMA_F16``", "Instr", "Number of VALU F16 FMA or multiply-add instructions issued"
  "``SQ_INSTS_VALU_TRANS_F16``", "Instr", "Number of VALU F16 Transcendental instructions issued"
  "``SQ_INSTS_VALU_ADD_F32``", "Instr", "Number of VALU full-precision floating-point (F32) ``ADD`` or ``SUB`` instructions issued"
  "``SQ_INSTS_VALU_MUL_F32``", "Instr", "Number of VALU F32 Multiply instructions issued"
  "``SQ_INSTS_VALU_FMA_F32``", "Instr", "Number of VALU F32 FMAor multiply-add instructions issued"
  "``SQ_INSTS_VALU_TRANS_F32``", "Instr", "Number of VALU F32 Transcendental instructions issued"
  "``SQ_INSTS_VALU_ADD_F64``", "Instr", "Number of VALU F64 ``ADD`` or ``SUB`` instructions issued"
  "``SQ_INSTS_VALU_MUL_F64``", "Instr", "Number of VALU F64 Multiply instructions issued"
  "``SQ_INSTS_VALU_FMA_F64``", "Instr", "Number of VALU F64 FMA or multiply-add instructions issued"
  "``SQ_INSTS_VALU_TRANS_F64``", "Instr", "Number of VALU F64 Transcendental instructions issued"
  "``SQ_INSTS_VALU_INT32``", "Instr", "Number of VALU 32-bit integer instructions (signed or unsigned) issued"
  "``SQ_INSTS_VALU_INT64``", "Instr", "Number of VALU 64-bit integer instructions (signed or unsigned) issued"
  "``SQ_INSTS_VALU_CVT``", "Instr", "Number of VALU Conversion instructions issued"
  "``SQ_INSTS_VALU_MFMA_I8``", "Instr", "Number of 8-bit Integer matrix FMA instructions issued"
  "``SQ_INSTS_VALU_MFMA_F16``", "Instr", "Number of F16 matrix FMA instructions issued"
  "``SQ_INSTS_VALU_MFMA_F32``", "Instr", "Number of F32 matrix FMA instructions issued"
  "``SQ_INSTS_VALU_MFMA_F64``", "Instr", "Number of F64 matrix FMA instructions issued"
  "``SQ_INSTS_MFMA``", "Instr", "Number of matrix FMA instructions issued"
  "``SQ_INSTS_VMEM_WR``", "Instr", "Number of vector memory write instructions (including flat) issued"
  "``SQ_INSTS_VMEM_RD``", "Instr", "Number of vector memory read instructions (including flat) issued"
  "``SQ_INSTS_VMEM``", "Instr", "Number of vector memory instructions issued, including both flat and buffer instructions"
  "``SQ_INSTS_SALU``", "Instr", "Number of scalar arithmetic logic unit (SALU) instructions issued"
  "``SQ_INSTS_SMEM``", "Instr", "Number of scalar memory instructions issued"
  "``SQ_INSTS_SMEM_NORM``", "Instr", "Number of scalar memory instructions normalized to match ``smem_level`` issued"
  "``SQ_INSTS_FLAT``", "Instr", "Number of flat instructions issued"
  "``SQ_INSTS_FLAT_LDS_ONLY``", "Instr", "**MI200 series only** Number of FLAT instructions that read/write only from/to LDS issued. Works only if ``EARLY_TA_DONE`` is enabled."
  "``SQ_INSTS_LDS``", "Instr", "Number of LDS instructions issued **(MI200: includes flat; MI300: does not include flat)**"
  "``SQ_INSTS_GDS``", "Instr", "Number of global data share instructions issued"
  "``SQ_INSTS_EXP_GDS``", "Instr", "Number of EXP and global data share instructions excluding skipped export instructions issued"
  "``SQ_INSTS_BRANCH``", "Instr", "Number of branch instructions issued"
  "``SQ_INSTS_SENDMSG``", "Instr", "Number of ``SENDMSG`` instructions including ``s_endpgm`` issued"
  "``SQ_INSTS_VSKIPPED``", "Instr", "Number of vector instructions skipped"

Flat instructions allow read, write, and atomic access to a generic memory address pointer that can
resolve to any of the following physical memories:

* Global Memory
* Scratch ("private")
* LDS ("shared")
* Invalid - ``MEM_VIOL`` TrapStatus

Matrix fused multiply-add operation counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_INSTS_VALU_MFMA_MOPS_I8``", "IOP", "Number of 8-bit integer matrix FMA ops in the unit of 512"
  "``SQ_INSTS_VALU_MFMA_MOPS_F16``", "FLOP", "Number of F16 floating matrix FMA ops in the unit of 512"
  "``SQ_INSTS_VALU_MFMA_MOPS_BF16``", "FLOP", "Number of BF16 floating matrix FMA ops in the unit of 512"
  "``SQ_INSTS_VALU_MFMA_MOPS_F32``", "FLOP", "Number of F32 floating matrix FMA ops in the unit of 512"
  "``SQ_INSTS_VALU_MFMA_MOPS_F64``", "FLOP", "Number of F64 floating matrix FMA ops in the unit of 512"

Level counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

  All level counters must be followed by ``SQ_ACCUM_PREV_HIRES`` counter to measure average latency.

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_ACCUM_PREV``", "Count", "Accumulated counter sample value where accumulation takes place once every four cycles"
  "``SQ_ACCUM_PREV_HIRES``", "Count", "Accumulated counter sample value where accumulation takes place once every cycle"
  "``SQ_LEVEL_WAVES``", "Waves", "Number of inflight waves"
  "``SQ_INST_LEVEL_VMEM``", "Instr", "Number of inflight vector memory (including flat) instructions"
  "``SQ_INST_LEVEL_SMEM``", "Instr", "Number of inflight scalar memory instructions"
  "``SQ_INST_LEVEL_LDS``", "Instr", "Number of inflight LDS (including flat) instructions"
  "``SQ_IFETCH_LEVEL``", "Instr", "Number of inflight instruction fetch requests from the cache"

Use the following formulae to calculate latencies:

* Vector memory latency = ``SQ_ACCUM_PREV_HIRES`` divided by ``SQ_INSTS_VMEM``
* Wave latency = ``SQ_ACCUM_PREV_HIRES`` divided by ``SQ_WAVE``
* LDS latency = ``SQ_ACCUM_PREV_HIRES`` divided by ``SQ_INSTS_LDS``
* Scalar memory latency = ``SQ_ACCUM_PREV_HIRES`` divided by ``SQ_INSTS_SMEM_NORM``
* Instruction fetch latency = ``SQ_ACCUM_PREV_HIRES`` divided by ``SQ_IFETCH``

Wavefront counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_WAVES``", "Waves", "Number of wavefronts dispatched to sequencers, including both new and restored wavefronts"
  "``SQ_WAVES_SAVED``", "Waves", "Number of context-saved waves"
  "``SQ_WAVES_RESTORED``", "Waves", "Number of context-restored waves sent to sequencers"
  "``SQ_WAVES_EQ_64``", "Waves", "Number of wavefronts with exactly 64 active threads sent to sequencers"
  "``SQ_WAVES_LT_64``", "Waves", "Number of wavefronts with less than 64 active threads sent to sequencers"
  "``SQ_WAVES_LT_48``", "Waves", "Number of wavefronts with less than 48 active threads sent to sequencers"
  "``SQ_WAVES_LT_32``", "Waves", "Number of wavefronts with less than 32 active threads sent to sequencers"
  "``SQ_WAVES_LT_16``", "Waves", "Number of wavefronts with less than 16 active threads sent to sequencers"

Wavefront cycle counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_CYCLES``", "Cycles", "Clock cycles"
  "``SQ_BUSY_CYCLES``", "Cycles", "Number of cycles while sequencers reports it to be busy"
  "``SQ_BUSY_CU_CYCLES``", "Qcycles", "Number of quad-cycles each compute unit is busy"
  "``SQ_VALU_MFMA_BUSY_CYCLES``", "Cycles", "Number of cycles the matrix FMA arithmetic logic unit (ALU) is busy"
  "``SQ_WAVE_CYCLES``", "Qcycles", "Number of quad-cycles spent by waves in the compute units"
  "``SQ_WAIT_ANY``", "Qcycles", "Number of quad-cycles spent waiting for anything"
  "``SQ_WAIT_INST_ANY``", "Qcycles", "Number of quad-cycles spent waiting for any instruction to be issued"
  "``SQ_ACTIVE_INST_ANY``", "Qcycles", "Number of quad-cycles spent by each wave to work on an instruction"
  "``SQ_ACTIVE_INST_VMEM``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on a vector memory instruction"
  "``SQ_ACTIVE_INST_LDS``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on an LDS instruction"
  "``SQ_ACTIVE_INST_VALU``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on a VALU instruction"
  "``SQ_ACTIVE_INST_SCA``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on a SALU or scalar memory instruction"
  "``SQ_ACTIVE_INST_EXP_GDS``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on an ``EXPORT`` or ``GDS`` instruction"
  "``SQ_ACTIVE_INST_MISC``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on a ``BRANCH`` or ``SENDMSG`` instruction"
  "``SQ_ACTIVE_INST_FLAT``", "Qcycles", "Number of quad-cycles spent by the sequencer instruction arbiter to work on a flat instruction"
  "``SQ_INST_CYCLES_VMEM_WR``", "Qcycles", "Number of quad-cycles spent to send addr and cmd data for vector memory write instructions"
  "``SQ_INST_CYCLES_VMEM_RD``", "Qcycles", "Number of quad-cycles spent to send addr and cmd data for vector memory read instructions"
  "``SQ_INST_CYCLES_SMEM``", "Qcycles", "Number of quad-cycles spent to execute scalar memory reads"
  "``SQ_INST_CYCLES_SALU``", "Qcycles", "Number of quad-cycles spent to execute non-memory read scalar operations"
  "``SQ_THREAD_CYCLES_VALU``", "Qcycles", "Number of quad-cycles spent to execute VALU operations on active threads"
  "``SQ_WAIT_INST_LDS``", "Qcycles", "Number of quad-cycles spent waiting for LDS instruction to be issued"

``SQ_THREAD_CYCLES_VALU`` is similar to ``INST_CYCLES_VALU``, but it's multiplied by the number of
active threads.

LDS counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_LDS_ATOMIC_RETURN``", "Cycles", "Number of atomic return cycles in LDS"
  "``SQ_LDS_BANK_CONFLICT``", "Cycles", "Number of cycles LDS is stalled by bank conflicts"
  "``SQ_LDS_ADDR_CONFLICT``", "Cycles", "Number of cycles LDS is stalled by address conflicts"
  "``SQ_LDS_UNALIGNED_STALL``", "Cycles", "Number of cycles LDS is stalled processing flat unaligned load or store operations"
  "``SQ_LDS_MEM_VIOLATIONS``", "Count", "Number of threads that have a memory violation in the LDS"
  "``SQ_LDS_IDX_ACTIVE``", "Cycles", "Number of cycles LDS is used for indexed operations"

Miscellaneous counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQ_IFETCH``", "Count", "Number of instruction fetch requests from L1i, in 32-byte width"
  "``SQ_ITEMS``", "Threads", "Number of valid items per wave"

.. _l1i-and-sl1d-cache-counters:

L1 instruction cache (L1i) and scalar L1 data cache (L1d) counters
---------------------------------------------------------------------------------------------------------------

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition"

  "``SQC_ICACHE_REQ``", "Req", "Number of L1 instruction (L1i) cache requests"
  "``SQC_ICACHE_HITS``", "Count", "Number of L1i cache hits"
  "``SQC_ICACHE_MISSES``", "Count", "Number of non-duplicate L1i cache misses including uncached requests"
  "``SQC_ICACHE_MISSES_DUPLICATE``", "Count", "Number of duplicate L1i cache misses whose previous lookup miss on the same cache line is not fulfilled yet"
  "``SQC_DCACHE_REQ``", "Req", "Number of scalar L1d requests"
  "``SQC_DCACHE_INPUT_VALID_READYB``", "Cycles", "Number of cycles while sequencer input is valid but scalar L1d is not ready"
  "``SQC_DCACHE_HITS``", "Count", "Number of scalar L1d hits"
  "``SQC_DCACHE_MISSES``", "Count", "Number of non-duplicate scalar L1d misses including uncached requests"
  "``SQC_DCACHE_MISSES_DUPLICATE``", "Count", "Number of duplicate scalar L1d misses"
  "``SQC_DCACHE_REQ_READ_1``", "Req", "Number of constant cache read requests in a single 32-bit data word"
  "``SQC_DCACHE_REQ_READ_2``", "Req", "Number of constant cache read requests in two 32-bit data words"
  "``SQC_DCACHE_REQ_READ_4``", "Req", "Number of constant cache read requests in four 32-bit data words"
  "``SQC_DCACHE_REQ_READ_8``", "Req", "Number of constant cache read requests in eight 32-bit data words"
  "``SQC_DCACHE_REQ_READ_16``", "Req", "Number of constant cache read requests in 16 32-bit data words"
  "``SQC_DCACHE_ATOMIC``", "Req", "Number of atomic requests"
  "``SQC_TC_REQ``", "Req", "Number of texture cache requests that were issued by instruction and constant caches"
  "``SQC_TC_INST_REQ``", "Req", "Number of instruction requests to the L2 cache"
  "``SQC_TC_DATA_READ_REQ``", "Req", "Number of data Read requests to the L2 cache"
  "``SQC_TC_DATA_WRITE_REQ``", "Req", "Number of data write requests to the L2 cache"
  "``SQC_TC_DATA_ATOMIC_REQ``", "Req", "Number of data atomic requests to the L2 cache"
  "``SQC_TC_STALL``", "Cycles", "Number of cycles while the valid requests to the L2 cache are stalled"

.. _vector-l1-cache-subsystem-counters:

Vector L1 cache subsystem counters
---------------------------------------------------------------------------------------------------------------

The vector L1 cache subsystem counters are further classified into texture addressing unit, texture data
unit, vector L1d or texture cache per pipe, and texture cache arbiter counters.

Texture addressing unit counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

  "``TA_TA_BUSY[n]``", "Cycles", "Texture addressing unit busy cycles", "0-15"
  "``TA_TOTAL_WAVEFRONTS[n]``", "Instr", "Number of wavefronts processed by texture addressing unit", "0-15"
  "``TA_BUFFER_WAVEFRONTS[n]``", "Instr", "Number of buffer wavefronts processed by texture addressing unit", "0-15"
  "``TA_BUFFER_READ_WAVEFRONTS[n]``", "Instr", "Number of buffer read wavefronts processed by texture addressing unit", "0-15"
  "``TA_BUFFER_WRITE_WAVEFRONTS[n]``", "Instr", "Number of buffer write wavefronts processed by texture addressing unit", "0-15"
  "``TA_BUFFER_ATOMIC_WAVEFRONTS[n]``", "Instr", "Number of buffer atomic wavefronts processed by texture addressing unit", "0-15"
  "``TA_BUFFER_TOTAL_CYCLES[n]``", "Cycles", "Number of buffer cycles (including read and write) issued to texture cache", "0-15"
  "``TA_BUFFER_COALESCED_READ_CYCLES[n]``", "Cycles", "Number of coalesced buffer read cycles issued to texture cache", "0-15"
  "``TA_BUFFER_COALESCED_WRITE_CYCLES[n]``", "Cycles", "Number of coalesced buffer write cycles issued to texture cache", "0-15"
  "``TA_ADDR_STALLED_BY_TC_CYCLES[n]``", "Cycles", "Number of cycles texture addressing unit address path is stalled by texture cache", "0-15"
  "``TA_DATA_STALLED_BY_TC_CYCLES[n]``", "Cycles", "Number of cycles texture addressing unit data path is stalled by texture cache", "0-15"
  "``TA_ADDR_STALLED_BY_TD_CYCLES[n]``", "Cycles", "Number of cycles texture addressing unit address path is stalled by texture data unit", "0-15"
  "``TA_FLAT_WAVEFRONTS[n]``", "Instr", "Number of flat opcode wavefronts processed by texture addressing unit", "0-15"
  "``TA_FLAT_READ_WAVEFRONTS[n]``", "Instr", "Number of flat opcode read wavefronts processed by texture addressing unit", "0-15"
  "``TA_FLAT_WRITE_WAVEFRONTS[n]``", "Instr", "Number of flat opcode write wavefronts processed by texture addressing unit", "0-15"
  "``TA_FLAT_ATOMIC_WAVEFRONTS[n]``", "Instr", "Number of flat opcode atomic wavefronts processed by texture addressing unit", "0-15"

Texture data unit counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

  "``TD_TD_BUSY[n]``", "Cycle", "Texture data unit busy cycles while it is processing or waiting for data", "0-15"
  "``TD_TC_STALL[n]``", "Cycle", "Number of cycles texture data unit is stalled waiting for texture cache data", "0-15"
  "``TD_SPI_STALL[n]``", "Cycle", "Number of cycles texture data unit is stalled by shader processor input", "0-15"
  "``TD_LOAD_WAVEFRONT[n]``", "Instr", "Number of wavefront instructions (read, write, atomic)", "0-15"
  "``TD_STORE_WAVEFRONT[n]``", "Instr", "Number of write wavefront instructions", "0-15"
  "``TD_ATOMIC_WAVEFRONT[n]``", "Instr", "Number of atomic wavefront instructions", "0-15"
  "``TD_COALESCABLE_WAVEFRONT[n]``", "Instr", "Number of coalescable wavefronts according to texture addressing unit", "0-15"

Texture cache per pipe counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

  "``TCP_GATE_EN1[n]``", "Cycles", "Number of cycles vector L1d interface clocks are turned on", "0-15"
  "``TCP_GATE_EN2[n]``", "Cycles", "Number of cycles vector L1d core clocks are turned on", "0-15"
  "``TCP_TD_TCP_STALL_CYCLES[n]``", "Cycles", "Number of cycles texture data unit stalls vector L1d", "0-15"
  "``TCP_TCR_TCP_STALL_CYCLES[n]``", "Cycles", "Number of cycles texture cache router stalls vector L1d", "0-15"
  "``TCP_READ_TAGCONFLICT_STALL_CYCLES[n]``", "Cycles", "Number of cycles tag RAM conflict stalls on a read", "0-15"
  "``TCP_WRITE_TAGCONFLICT_STALL_CYCLES[n]``", "Cycles", "Number of cycles tag RAM conflict stalls on a write", "0-15"
  "``TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES[n]``", "Cycles", "Number of cycles tag RAM conflict stalls on an atomic", "0-15"
  "``TCP_PENDING_STALL_CYCLES[n]``", "Cycles", "Number of cycles vector L1d is stalled due to data pending from L2 Cache", "0-15"
  "``TCP_TCP_TA_DATA_STALL_CYCLES``", "Cycles", "Number of cycles texture cache per pipe stalls texture addressing unit data interface", "NA"
  "``TCP_TA_TCP_STATE_READ[n]``", "Req", "Number of state reads", "0-15"
  "``TCP_VOLATILE[n]``", "Req", "Number of L1 volatile pixels or buffers from texture addressing unit", "0-15"
  "``TCP_TOTAL_ACCESSES[n]``", "Req", "Number of vector L1d accesses. Equals ``TCP_PERF_SEL_TOTAL_READ`+`TCP_PERF_SEL_TOTAL_NONREAD``", "0-15"
  "``TCP_TOTAL_READ[n]``", "Req", "Number of vector L1d read accesses", "0-15"
  "``TCP_TOTAL_WRITE[n]``", "Req", "Number of vector L1d write accesses", "0-15"
  "``TCP_TOTAL_ATOMIC_WITH_RET[n]``", "Req", "Number of vector L1d atomic requests with return", "0-15"
  "``TCP_TOTAL_ATOMIC_WITHOUT_RET[n]``", "Req", "Number of vector L1d atomic without return", "0-15"
  "``TCP_TOTAL_WRITEBACK_INVALIDATES[n]``", "Count", "Total number of vector L1d writebacks and invalidates", "0-15"
  "``TCP_UTCL1_REQUEST[n]``", "Req", "Number of address translation requests to unified translation cache (L1)", "0-15"
  "``TCP_UTCL1_TRANSLATION_HIT[n]``", "Req", "Number of unified translation cache (L1) translation hits", "0-15"
  "``TCP_UTCL1_TRANSLATION_MISS[n]``", "Req", "Number of unified translation cache (L1) translation misses", "0-15"
  "``TCP_UTCL1_PERMISSION_MISS[n]``", "Req", "Number of unified translation cache (L1) permission misses", "0-15"
  "``TCP_TOTAL_CACHE_ACCESSES[n]``", "Req", "Number of vector L1d cache accesses including hits and misses", "0-15"
  "``TCP_TCP_LATENCY[n]``", "Cycles", "**MI200 series only** Accumulated wave access latency to vL1D over all wavefronts", "0-15"
  "``TCP_TCC_READ_REQ_LATENCY[n]``", "Cycles", "**MI200 series only** Total vL1D to L2 request latency over all wavefronts for reads and atomics with return", "0-15"
  "``TCP_TCC_WRITE_REQ_LATENCY[n]``", "Cycles", "**MI200 series only** Total vL1D to L2 request latency over all wavefronts for writes and atomics without return", "0-15"
  "``TCP_TCC_READ_REQ[n]``", "Req", "Number of read requests to L2 cache", "0-15"
  "``TCP_TCC_WRITE_REQ[n]``", "Req", "Number of write requests to L2 cache", "0-15"
  "``TCP_TCC_ATOMIC_WITH_RET_REQ[n]``", "Req", "Number of atomic requests to L2 cache with return", "0-15"
  "``TCP_TCC_ATOMIC_WITHOUT_RET_REQ[n]``", "Req", "Number of atomic requests to L2 cache without return", "0-15"
  "``TCP_TCC_NC_READ_REQ[n]``", "Req", "Number of non-coherently cached read requests to L2 cache", "0-15"
  "``TCP_TCC_UC_READ_REQ[n]``", "Req", "Number of uncached read requests to L2 cache", "0-15"
  "``TCP_TCC_CC_READ_REQ[n]``", "Req", "Number of coherently cached read requests to L2 cache", "0-15"
  "``TCP_TCC_RW_READ_REQ[n]``", "Req", "Number of coherently cached with write read requests to L2 cache", "0-15"
  "``TCP_TCC_NC_WRITE_REQ[n]``", "Req", "Number of non-coherently cached write requests to L2 cache", "0-15"
  "``TCP_TCC_UC_WRITE_REQ[n]``", "Req", "Number of uncached write requests to L2 cache", "0-15"
  "``TCP_TCC_CC_WRITE_REQ[n]``", "Req", "Number of coherently cached write requests to L2 cache", "0-15"
  "``TCP_TCC_RW_WRITE_REQ[n]``", "Req", "Number of coherently cached with write write requests to L2 cache", "0-15"
  "``TCP_TCC_NC_ATOMIC_REQ[n]``", "Req", "Number of non-coherently cached atomic requests to L2 cache", "0-15"
  "``TCP_TCC_UC_ATOMIC_REQ[n]``", "Req", "Number of uncached atomic requests to L2 cache", "0-15"
  "``TCP_TCC_CC_ATOMIC_REQ[n]``", "Req", "Number of coherently cached atomic requests to L2 cache", "0-15"
  "``TCP_TCC_RW_ATOMIC_REQ[n]``", "Req", "Number of coherently cached with write atomic requests to L2 cache", "0-15"

Note that:

* ``TCP_TOTAL_READ[n]`` = ``TCP_PERF_SEL_TOTAL_HIT_LRU_READ`` + ``TCP_PERF_SEL_TOTAL_MISS_LRU_READ`` + ``TCP_PERF_SEL_TOTAL_MISS_EVICT_READ``
* ``TCP_TOTAL_WRITE[n]`` = ``TCP_PERF_SEL_TOTAL_MISS_LRU_WRITE``+ ``TCP_PERF_SEL_TOTAL_MISS_EVICT_WRITE``
* ``TCP_TOTAL_WRITEBACK_INVALIDATES[n]`` = ``TCP_PERF_SEL_TOTAL_WBINVL1``+ ``TCP_PERF_SEL_TOTAL_WBINVL1_VOL``+ ``TCP_PERF_SEL_CP_TCP_INVALIDATE``+ ``TCP_PERF_SEL_SQ_TCP_INVALIDATE_VOL``

Texture cache arbiter counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

  "``TCA_CYCLE[n]``", "Cycles", "Number of texture cache arbiter cycles", "0-31"
  "``TCA_BUSY[n]``", "Cycles", "Number of cycles texture cache arbiter has a pending request", "0-31"

.. _l2-cache-access-counters:

L2 cache access counters
---------------------------------------------------------------------------------------------------------------

L2 cache is also known as texture cache per channel.

.. tab-set::

    .. tab-item:: MI300 hardware counter

      .. csv-table::
        :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

        "``TCC_CYCLE[n]``", "Cycles", "Number of L2 cache free-running clocks", "0-31"
        "``TCC_BUSY[n]``", "Cycles", "Number of L2 cache busy cycles", "0-31"
        "``TCC_REQ[n]``", "Req", "Number of L2 cache requests of all types (measured at the tag block)", "0-31"
        "``TCC_STREAMING_REQ[n]``", "Req", "Number of L2 cache streaming requests (measured at the tag block)", "0-31"
        "``TCC_NC_REQ[n]``", "Req", "Number of non-coherently cached requests (measured at the tag block)", "0-31"
        "``TCC_UC_REQ[n]``", "Req", "Number of uncached requests. This is measured at the tag block", "0-31"
        "``TCC_CC_REQ[n]``", "Req", "Number of coherently cached requests. This is measured at the tag block", "0-31"
        "``TCC_RW_REQ[n]``", "Req", "Number of coherently cached with write requests. This is measured at the tag block", "0-31"
        "``TCC_PROBE[n]``", "Req", "Number of probe requests", "0-31"
        "``TCC_PROBE_ALL[n]``", "Req", "Number of external probe requests with ``EA_TCC_preq_all == 1``", "0-31"
        "``TCC_READ[n]``", "Req", "Number of L2 cache read requests (includes compressed reads but not metadata reads)", "0-31"
        "``TCC_WRITE[n]``", "Req", "Number of L2 cache write requests", "0-31"
        "``TCC_ATOMIC[n]``", "Req", "Number of L2 cache atomic requests of all types", "0-31"
        "``TCC_HIT[n]``", "Req", "Number of L2 cache hits", "0-31"
        "``TCC_MISS[n]``", "Req", "Number of L2 cache misses", "0-31"
        "``TCC_WRITEBACK[n]``", "Req", "Number of lines written back to the main memory, including writebacks of dirty lines and uncached write or atomic requests", "0-31"
        "``TCC_EA0_WRREQ[n]``", "Req", "Number of 32-byte and 64-byte transactions going over the ``TC_EA_wrreq`` interface (doesn't include probe commands)", "0-31"
        "``TCC_EA0_WRREQ_64B[n]``", "Req", "Total number of 64-byte transactions (write or ``CMPSWAP``) going over the ``TC_EA_wrreq`` interface", "0-31"
        "``TCC_EA0_WR_UNCACHED_32B[n]``", "Req", "Number of 32 or 64-byte write or atomic going over the ``TC_EA_wrreq`` interface due to uncached traffic", "0-31"
        "``TCC_EA0_WRREQ_STALL[n]``", "Cycles", "Number of cycles a write request is stalled", "0-31"
        "``TCC_EA0_WRREQ_IO_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of input-output (IO) credits", "0-31"
        "``TCC_EA0_WRREQ_GMI_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of GMI credits", "0-31"
        "``TCC_EA0_WRREQ_DRAM_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of DRAM credits", "0-31"
        "``TCC_TOO_MANY_EA_WRREQS_STALL[n]``", "Cycles", "Number of cycles the L2 cache is unable to send an efficiency arbiter write request due to it reaching its maximum capacity of pending efficiency arbiter write requests", "0-31"
        "``TCC_EA0_WRREQ_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter write requests in flight", "0-31"
        "``TCC_EA0_ATOMIC[n]``", "Req", "Number of 32-byte or 64-byte atomic requests going over the ``TC_EA_wrreq`` interface", "0-31"
        "``TCC_EA0_ATOMIC_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter atomic requests in flight", "0-31"
        "``TCC_EA0_RDREQ[n]``", "Req", "Number of 32-byte or 64-byte read requests to efficiency arbiter", "0-31"
        "``TCC_EA0_RDREQ_32B[n]``", "Req", "Number of 32-byte read requests to efficiency arbiter", "0-31"
        "``TCC_EA0_RD_UNCACHED_32B[n]``", "Req", "Number of 32-byte efficiency arbiter reads due to uncached traffic. A 64-byte request is counted as 2", "0-31"
        "``TCC_EA0_RDREQ_IO_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of IO credits", "0-31"
        "``TCC_EA0_RDREQ_GMI_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of GMI credits", "0-31"
        "``TCC_EA0_RDREQ_DRAM_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of DRAM credits", "0-31"
        "``TCC_EA0_RDREQ_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter read requests in flight", "0-31"
        "``TCC_EA0_RDREQ_DRAM[n]``", "Req", "Number of 32-byte or 64-byte efficiency arbiter read requests to High Bandwidth Memory (HBM)", "0-31"
        "``TCC_EA0_WRREQ_DRAM[n]``", "Req", "Number of 32-byte or 64-byte efficiency arbiter write requests to HBM", "0-31"
        "``TCC_TAG_STALL[n]``", "Cycles", "Number of cycles the normal request pipeline in the tag is stalled for any reason", "0-31"
        "``TCC_NORMAL_WRITEBACK[n]``", "Req", "Number of writebacks due to requests that are not writeback requests", "0-31"
        "``TCC_ALL_TC_OP_WB_WRITEBACK[n]``", "Req", "Number of writebacks due to all ``TC_OP`` writeback requests", "0-31"
        "``TCC_NORMAL_EVICT[n]``", "Req", "Number of evictions due to requests that are not invalidate or probe requests", "0-31"
        "``TCC_ALL_TC_OP_INV_EVICT[n]``", "Req", "Number of evictions due to all ``TC_OP`` invalidate requests", "0-31"

    .. tab-item:: MI200 hardware counter

      .. csv-table::
        :header: "Hardware counter", "Unit", "Definition", "Value range for ``n``"

        "``TCC_CYCLE[n]``", "Cycles", "Number of L2 cache free-running clocks", "0-31"
        "``TCC_BUSY[n]``", "Cycles", "Number of L2 cache busy cycles", "0-31"
        "``TCC_REQ[n]``", "Req", "Number of L2 cache requests of all types (measured at the tag block)", "0-31"
        "``TCC_STREAMING_REQ[n]``", "Req", "Number of L2 cache streaming requests (measured at the tag block)", "0-31"
        "``TCC_NC_REQ[n]``", "Req", "Number of non-coherently cached requests (measured at the tag block)", "0-31"
        "``TCC_UC_REQ[n]``", "Req", "Number of uncached requests. This is measured at the tag block", "0-31"
        "``TCC_CC_REQ[n]``", "Req", "Number of coherently cached requests. This is measured at the tag block", "0-31"
        "``TCC_RW_REQ[n]``", "Req", "Number of coherently cached with write requests. This is measured at the tag block", "0-31"
        "``TCC_PROBE[n]``", "Req", "Number of probe requests", "0-31"
        "``TCC_PROBE_ALL[n]``", "Req", "Number of external probe requests with ``EA_TCC_preq_all == 1``", "0-31"
        "``TCC_READ[n]``", "Req", "Number of L2 cache read requests (includes compressed reads but not metadata reads)", "0-31"
        "``TCC_WRITE[n]``", "Req", "Number of L2 cache write requests", "0-31"
        "``TCC_ATOMIC[n]``", "Req", "Number of L2 cache atomic requests of all types", "0-31"
        "``TCC_HIT[n]``", "Req", "Number of L2 cache hits", "0-31"
        "``TCC_MISS[n]``", "Req", "Number of L2 cache misses", "0-31"
        "``TCC_WRITEBACK[n]``", "Req", "Number of lines written back to the main memory, including writebacks of dirty lines and uncached write or atomic requests", "0-31"
        "``TCC_EA_WRREQ[n]``", "Req", "Number of 32-byte and 64-byte transactions going over the ``TC_EA_wrreq`` interface (doesn't include probe commands)", "0-31"
        "``TCC_EA_WRREQ_64B[n]``", "Req", "Total number of 64-byte transactions (write or ``CMPSWAP``) going over the ``TC_EA_wrreq`` interface", "0-31"
        "``TCC_EA_WR_UNCACHED_32B[n]``", "Req", "Number of 32 write or atomic going over the ``TC_EA_wrreq`` interface due to uncached traffic. A 64-byte request will be counted as 2", "0-31"
        "``TCC_EA_WRREQ_STALL[n]``", "Cycles", "Number of cycles a write request is stalled", "0-31"
        "``TCC_EA_WRREQ_IO_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of input-output (IO) credits", "0-31"
        "``TCC_EA_WRREQ_GMI_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of GMI credits", "0-31"
        "``TCC_EA_WRREQ_DRAM_CREDIT_STALL[n]``", "Cycles", "Number of cycles an efficiency arbiter write request is stalled due to the interface running out of DRAM credits", "0-31"
        "``TCC_TOO_MANY_EA_WRREQS_STALL[n]``", "Cycles", "Number of cycles the L2 cache is unable to send an efficiency arbiter write request due to it reaching its maximum capacity of pending efficiency arbiter write requests", "0-31"
        "``TCC_EA_WRREQ_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter write requests in flight", "0-31"
        "``TCC_EA_ATOMIC[n]``", "Req", "Number of 32-byte or 64-byte atomic requests going over the ``TC_EA_wrreq`` interface", "0-31"
        "``TCC_EA_ATOMIC_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter atomic requests in flight", "0-31"
        "``TCC_EA_RDREQ[n]``", "Req", "Number of 32-byte or 64-byte read requests to efficiency arbiter", "0-31"
        "``TCC_EA_RDREQ_32B[n]``", "Req", "Number of 32-byte read requests to efficiency arbiter", "0-31"
        "``TCC_EA_RD_UNCACHED_32B[n]``", "Req", "Number of 32-byte efficiency arbiter reads due to uncached traffic. A 64-byte request is counted as 2", "0-31"
        "``TCC_EA_RDREQ_IO_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of IO credits", "0-31"
        "``TCC_EA_RDREQ_GMI_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of GMI credits", "0-31"
        "``TCC_EA_RDREQ_DRAM_CREDIT_STALL[n]``", "Cycles", "Number of cycles there is a stall due to the read request interface running out of DRAM credits", "0-31"
        "``TCC_EA_RDREQ_LEVEL[n]``", "Req", "The accumulated number of efficiency arbiter read requests in flight", "0-31"
        "``TCC_EA_RDREQ_DRAM[n]``", "Req", "Number of 32-byte or 64-byte efficiency arbiter read requests to High Bandwidth Memory (HBM)", "0-31"
        "``TCC_EA_WRREQ_DRAM[n]``", "Req", "Number of 32-byte or 64-byte efficiency arbiter write requests to HBM", "0-31"
        "``TCC_TAG_STALL[n]``", "Cycles", "Number of cycles the normal request pipeline in the tag is stalled for any reason", "0-31"
        "``TCC_NORMAL_WRITEBACK[n]``", "Req", "Number of writebacks due to requests that are not writeback requests", "0-31"
        "``TCC_ALL_TC_OP_WB_WRITEBACK[n]``", "Req", "Number of writebacks due to all ``TC_OP`` writeback requests", "0-31"
        "``TCC_NORMAL_EVICT[n]``", "Req", "Number of evictions due to requests that are not invalidate or probe requests", "0-31"
        "``TCC_ALL_TC_OP_INV_EVICT[n]``", "Req", "Number of evictions due to all ``TC_OP`` invalidate requests", "0-31"

Note the following:

* ``TCC_REQ[n]`` may be more than the number of requests arriving at the texture cache per channel,
  but it's a good indication of the total amount of work that needs to be performed.
* For ``TCC_EA0_WRREQ[n]``, atomics may travel over the same interface and are generally classified as
  write requests.
* CC mtypes can produce uncached requests, and those are included in
  ``TCC_EA0_WR_UNCACHED_32B[n]``
* ``TCC_EA0_WRREQ_LEVEL[n]`` is primarily intended to measure average efficiency arbiter write latency.

  * Average write latency = ``TCC_PERF_SEL_EA0_WRREQ_LEVEL`` divided by ``TCC_PERF_SEL_EA0_WRREQ``

* ``TCC_EA0_ATOMIC_LEVEL[n]`` is primarily intended to measure average efficiency arbiter atomic
  latency

  * Average atomic latency = ``TCC_PERF_SEL_EA0_WRREQ_ATOMIC_LEVEL`` divided by ``TCC_PERF_SEL_EA0_WRREQ_ATOMIC``

* ``TCC_EA0_RDREQ_LEVEL[n]`` is primarily intended to measure average efficiency arbiter read latency.

  * Average read latency = ``TCC_PERF_SEL_EA0_RDREQ_LEVEL`` divided by ``TCC_PERF_SEL_EA0_RDREQ``

* Stalls can occur regardless of the need for a read to be performed
* Normally, stalls are measured exactly at one point in the pipeline however in the case of
  ``TCC_TAG_STALL[n]``, probes can stall the pipeline at a variety of places. There is no single point that
  can accurately measure the total stalls

MI300 and MI200 series derived metrics list
==============================================================

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``ALUStalledByLDS``", "Percentage of GPU time ALU units are stalled due to the LDS input queue being full or the output queue not being ready (value range: 0% (optimal) to 100%)"
  "``FetchSize``", "Total kilobytes fetched from the video memory; measured with all extra fetches and any cache or memory effects taken into account"
  "``FlatLDSInsts``", "Average number of flat instructions that read from or write to LDS, run per work item (affected by flow control)"
  "``FlatVMemInsts``", "Average number of flat instructions that read from or write to the video memory, run per work item (affected by flow control). Includes flat instructions that read from or write to scratch"
  "``GDSInsts``", "Average number of global data share read or write instructions run per work item (affected by flow control)"
  "``GPUBusy``", "Percentage of time GPU is busy"
  "``L2CacheHit``", "Percentage of fetch, write, atomic, and other instructions that hit the data in L2 cache (value range: 0% (no hit) to 100% (optimal))"
  "``LDSBankConflict``", "Percentage of GPU time LDS is stalled by bank conflicts (value range: 0% (optimal) to 100%)"
  "``LDSInsts``", "Average number of LDS read or write instructions run per work item (affected by flow control). Excludes flat instructions that read from or write to LDS."
  "``MemUnitBusy``", "Percentage of GPU time the memory unit is active, which is measured with all extra fetches and writes and any cache or memory effects taken into account (value range: 0% to 100% (fetch-bound))"
  "``MemUnitStalled``", "Percentage of GPU time the memory unit is stalled (value range: 0% (optimal) to 100%)"
  "``MemWrites32B``", "Total number of effective 32B write transactions to the memory"
  "``TCA_BUSY_sum``", "Total number of cycles texture cache arbiter has a pending request, over all texture cache arbiter instances"
  "``TCA_CYCLE_sum``", "Total number of cycles over all texture cache arbiter instances"
  "``SALUBusy``", "Percentage of GPU time scalar ALU instructions are processed (value range: 0% to 100% (optimal))"
  "``SALUInsts``", "Average number of scalar ALU instructions run per work item (affected by flow control)"
  "``SFetchInsts``", "Average number of scalar fetch instructions from the video memory run per work item (affected by flow control)"
  "``VALUBusy``", "Percentage of GPU time vector ALU instructions are processed (value range: 0% to 100% (optimal))"
  "``VALUInsts``", "Average number of vector ALU instructions run per work item (affected by flow control)"
  "``VALUUtilization``", "Percentage of active vector ALU threads in a wave, where a lower number can mean either more thread divergence in a wave or that the work-group size is not a multiple of 64 (value range: 0%, 100% (optimal - no thread divergence))"
  "``VFetchInsts``", "Average number of vector fetch instructions from the video memory run per work-item (affected by flow control); excludes flat instructions that fetch from video memory"
  "``VWriteInsts``", "Average number of vector write instructions to the video memory run per work-item (affected by flow control); excludes flat instructions that write to video memory"
  "``Wavefronts``", "Total wavefronts"
  "``WRITE_REQ_32B``", "Total number of 32-byte effective memory writes"
  "``WriteSize``", "Total kilobytes written to the video memory; measured with all extra fetches and any cache or memory effects taken into account"
  "``WriteUnitStalled``", "Percentage of GPU time the write unit is stalled (value range: 0% (optimal) to 100%)"

You can lower ``ALUStalledByLDS`` by reducing LDS bank conflicts or number of LDS accesses.
You can lower ``MemUnitStalled`` by reducing the number or size of fetches and writes.
``MemUnitBusy`` includes the stall time (``MemUnitStalled``).

Hardware counters by and over all texture addressing unit instances
---------------------------------------------------------------------------------------------------------------

The following table shows the hardware counters *by* all texture addressing unit instances.

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TA_BUFFER_WAVEFRONTS_sum``", "Total number of buffer wavefronts processed"
  "``TA_BUFFER_READ_WAVEFRONTS_sum``", "Total number of buffer read wavefronts processed"
  "``TA_BUFFER_WRITE_WAVEFRONTS_sum``", "Total number of buffer write wavefronts processed"
  "``TA_BUFFER_ATOMIC_WAVEFRONTS_sum``", "Total number of buffer atomic wavefronts processed"
  "``TA_BUFFER_TOTAL_CYCLES_sum``", "Total number of buffer cycles (including read and write) issued to texture cache"
  "``TA_BUFFER_COALESCED_READ_CYCLES_sum``", "Total number of coalesced buffer read cycles issued to texture cache"
  "``TA_BUFFER_COALESCED_WRITE_CYCLES_sum``", "Total number of coalesced buffer write cycles issued to texture cache"
  "``TA_FLAT_READ_WAVEFRONTS_sum``", "Sum of flat opcode reads processed"
  "``TA_FLAT_WRITE_WAVEFRONTS_sum``", "Sum of flat opcode writes processed"
  "``TA_FLAT_WAVEFRONTS_sum``", "Total number of flat opcode wavefronts processed"
  "``TA_FLAT_ATOMIC_WAVEFRONTS_sum``", "Total number of flat opcode atomic wavefronts processed"
  "``TA_TOTAL_WAVEFRONTS_sum``", "Total number of wavefronts processed"

The following table shows the hardware counters *over* all texture addressing unit instances.

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TA_ADDR_STALLED_BY_TC_CYCLES_sum``", "Total number of cycles texture addressing unit address path is stalled by texture cache"
  "``TA_ADDR_STALLED_BY_TD_CYCLES_sum``", "Total number of cycles texture addressing unit address path is stalled by texture data unit"
  "``TA_BUSY_avr``", "Average number of busy cycles"
  "``TA_BUSY_max``", "Maximum number of texture addressing unit busy cycles"
  "``TA_BUSY_min``", "Minimum number of texture addressing unit busy cycles"
  "``TA_DATA_STALLED_BY_TC_CYCLES_sum``", "Total number of cycles texture addressing unit data path is stalled by texture cache"
  "``TA_TA_BUSY_sum``", "Total number of texture addressing unit busy cycles"

Hardware counters over all texture cache per channel instances
---------------------------------------------------------------------------------------------------------------

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TCC_ALL_TC_OP_WB_WRITEBACK_sum``", "Total number of writebacks due to all ``TC_OP`` writeback requests."
  "``TCC_ALL_TC_OP_INV_EVICT_sum``", "Total number of evictions due to all ``TC_OP`` invalidate requests."
  "``TCC_ATOMIC_sum``", "Total number of L2 cache atomic requests of all types."
  "``TCC_BUSY_avr``", "Average number of L2 cache busy cycles."
  "``TCC_BUSY_sum``", "Total number of L2 cache busy cycles."
  "``TCC_CC_REQ_sum``", "Total number of coherently cached requests."
  "``TCC_CYCLE_sum``", "Total number of L2 cache free running clocks."
  "``TCC_EA0_WRREQ_sum``", "Total number of 32-byte and 64-byte transactions going over the ``TC_EA0_wrreq`` interface. Atomics may travel over the same interface and are generally classified as write requests. This does not include probe commands."
  "``TCC_EA0_WRREQ_64B_sum``", "Total number of 64-byte transactions (write or `CMPSWAP`) going over the ``TC_EA0_wrreq`` interface."
  "``TCC_EA0_WR_UNCACHED_32B_sum``", "Total Number of 32-byte write or atomic going over the ``TC_EA0_wrreq`` interface due to uncached traffic. Note that coherently cached mtypes can produce uncached requests, and those are included in this. A 64-byte request is counted as 2."
  "``TCC_EA0_WRREQ_STALL_sum``", "Total Number of cycles a write request is stalled, over all instances."
  "``TCC_EA0_WRREQ_IO_CREDIT_STALL_sum``", "Total number of cycles an efficiency arbiter write request is stalled due to the interface running out of IO credits, over all instances."
  "``TCC_EA0_WRREQ_GMI_CREDIT_STALL_sum``", "Total number of cycles an efficiency arbiter write request is stalled due to the interface running out of GMI credits, over all instances."
  "``TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum``", "Total number of cycles an efficiency arbiter write request is stalled due to the interface running out of DRAM credits, over all instances."
  "``TCC_EA0_WRREQ_LEVEL_sum``", "Total number of efficiency arbiter write requests in flight."
  "``TCC_EA0_RDREQ_LEVEL_sum``", "Total number of efficiency arbiter read requests in flight."
  "``TCC_EA0_ATOMIC_sum``", "Total Number of 32-byte or 64-byte atomic requests going over the ``TC_EA0_wrreq`` interface."
  "``TCC_EA0_ATOMIC_LEVEL_sum``", "Total number of efficiency arbiter atomic requests in flight."
  "``TCC_EA0_RDREQ_sum``", "Total number of 32-byte or 64-byte read requests to efficiency arbiter."
  "``TCC_EA0_RDREQ_32B_sum``", "Total number of 32-byte read requests to efficiency arbiter."
  "``TCC_EA0_RD_UNCACHED_32B_sum``", "Total number of 32-byte efficiency arbiter reads due to uncached traffic."
  "``TCC_EA0_RDREQ_IO_CREDIT_STALL_sum``", "Total number of cycles there is a stall due to the read request interface running out of IO credits."
  "``TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum``", "Total number of cycles there is a stall due to the read request interface running out of GMI credits."
  "``TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum``", "Total number of cycles there is a stall due to the read request interface running out of DRAM credits."
  "``TCC_EA0_RDREQ_DRAM_sum``", "Total number of 32-byte or 64-byte efficiency arbiter read requests to HBM."
  "``TCC_EA0_WRREQ_DRAM_sum``", "Total number of 32-byte or 64-byte efficiency arbiter write requests to HBM."
  "``TCC_HIT_sum``", "Total number of L2 cache hits."
  "``TCC_MISS_sum``", "Total number of L2 cache misses."
  "``TCC_NC_REQ_sum``", "Total number of non-coherently cached requests."
  "``TCC_NORMAL_WRITEBACK_sum``", "Total number of writebacks due to requests that are not writeback requests."
  "``TCC_NORMAL_EVICT_sum``", "Total number of evictions due to requests that are not invalidate or probe requests."
  "``TCC_PROBE_sum``", "Total number of probe requests."
  "``TCC_PROBE_ALL_sum``", "Total number of external probe requests with ``EA0_TCC_preq_all == 1``."
  "``TCC_READ_sum``", "Total number of L2 cache read requests (including compressed reads but not metadata reads)."
  "``TCC_REQ_sum``", "Total number of all types of L2 cache requests."
  "``TCC_RW_REQ_sum``", "Total number of coherently cached with write requests."
  "``TCC_STREAMING_REQ_sum``", "Total number of L2 cache streaming requests."
  "``TCC_TAG_STALL_sum``", "Total number of cycles the normal request pipeline in the tag is stalled for any reason."
  "``TCC_TOO_MANY_EA0_WRREQS_STALL_sum``", "Total number of cycles L2 cache is unable to send an efficiency arbiter write request due to it reaching its maximum capacity of pending efficiency arbiter write requests."
  "``TCC_UC_REQ_sum``", "Total number of uncached requests."
  "``TCC_WRITE_sum``", "Total number of L2 cache write requests."
  "``TCC_WRITEBACK_sum``", "Total number of lines written back to the main memory including writebacks of dirty lines and uncached write or atomic requests."
  "``TCC_WRREQ_STALL_max``", "Maximum number of cycles a write request is stalled."

Hardware counters by, for, or over all texture cache per pipe instances
----------------------------------------------------------------------------------------------------------------

The following table shows the hardware counters *by* all texture cache per pipe instances.

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TCP_TA_TCP_STATE_READ_sum``", "Total number of state reads by ATCPPI"
  "``TCP_TOTAL_CACHE_ACCESSES_sum``", "Total number of vector L1d accesses (including hits and misses)"
  "``TCP_UTCL1_PERMISSION_MISS_sum``", "Total number of unified translation cache (L1) permission misses"
  "``TCP_UTCL1_REQUEST_sum``", "Total number of address translation requests to unified translation cache (L1)"
  "``TCP_UTCL1_TRANSLATION_MISS_sum``", "Total number of unified translation cache (L1) translation misses"
  "``TCP_UTCL1_TRANSLATION_HIT_sum``", "Total number of unified translation cache (L1) translation hits"

The following table shows the hardware counters *for* all texture cache per pipe instances.

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TCP_TCC_READ_REQ_LATENCY_sum``", "Total vector L1d to L2 request latency over all wavefronts for reads and atomics with return"
  "``TCP_TCC_WRITE_REQ_LATENCY_sum``", "Total vector L1d to L2 request latency over all wavefronts for writes and atomics without return"
  "``TCP_TCP_LATENCY_sum``", "Total wave access latency to vector L1d over all wavefronts"

The following table shows the hardware counters *over* all texture cache per pipe instances.

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TCP_ATOMIC_TAGCONFLICT_STALL_CYCLES_sum``", "Total number of cycles tag RAM conflict stalls on an atomic"
  "``TCP_GATE_EN1_sum``", "Total number of cycles vector L1d interface clocks are turned on"
  "``TCP_GATE_EN2_sum``", "Total number of cycles vector L1d core clocks are turned on"
  "``TCP_PENDING_STALL_CYCLES_sum``", "Total number of cycles vector L1d cache is stalled due to data pending from L2 Cache"
  "``TCP_READ_TAGCONFLICT_STALL_CYCLES_sum``", "Total number of cycles tag RAM conflict stalls on a read"
  "``TCP_TCC_ATOMIC_WITH_RET_REQ_sum``", "Total number of atomic requests to L2 cache with return"
  "``TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum``", "Total number of atomic requests to L2 cache without return"
  "``TCP_TCC_CC_READ_REQ_sum``", "Total number of coherently cached read requests to L2 cache"
  "``TCP_TCC_CC_WRITE_REQ_sum``", "Total number of coherently cached write requests to L2 cache"
  "``TCP_TCC_CC_ATOMIC_REQ_sum``", "Total number of coherently cached atomic requests to L2 cache"
  "``TCP_TCC_NC_READ_REQ_sum``", "Total number of non-coherently cached read requests to L2 cache"
  "``TCP_TCC_NC_WRITE_REQ_sum``", "Total number of non-coherently cached write requests to L2 cache"
  "``TCP_TCC_NC_ATOMIC_REQ_sum``", "Total number of non-coherently cached atomic requests to L2 cache"
  "``TCP_TCC_READ_REQ_sum``", "Total number of read requests to L2 cache"
  "``TCP_TCC_RW_READ_REQ_sum``", "Total number of coherently cached with write read requests to L2 cache"
  "``TCP_TCC_RW_WRITE_REQ_sum``", "Total number of coherently cached with write write requests to L2 cache"
  "``TCP_TCC_RW_ATOMIC_REQ_sum``", "Total number of coherently cached with write atomic requests to L2 cache"
  "``TCP_TCC_UC_READ_REQ_sum``", "Total number of uncached read requests to L2 cache"
  "``TCP_TCC_UC_WRITE_REQ_sum``", "Total number of uncached write requests to L2 cache"
  "``TCP_TCC_UC_ATOMIC_REQ_sum``", "Total number of uncached atomic requests to L2 cache"
  "``TCP_TCC_WRITE_REQ_sum``", "Total number of write requests to L2 cache"
  "``TCP_TCR_TCP_STALL_CYCLES_sum``", "Total number of cycles texture cache router stalls vector L1d"
  "``TCP_TD_TCP_STALL_CYCLES_sum``", "Total number of cycles texture data unit stalls vector L1d"
  "``TCP_TOTAL_ACCESSES_sum``", "Total number of vector L1d accesses"
  "``TCP_TOTAL_READ_sum``", "Total number of vector L1d read accesses"
  "``TCP_TOTAL_WRITE_sum``", "Total number of vector L1d write accesses"
  "``TCP_TOTAL_ATOMIC_WITH_RET_sum``", "Total number of vector L1d atomic requests with return"
  "``TCP_TOTAL_ATOMIC_WITHOUT_RET_sum``", "Total number of vector L1d atomic requests without return"
  "``TCP_TOTAL_WRITEBACK_INVALIDATES_sum``", "Total number of vector L1d writebacks and invalidates"
  "``TCP_VOLATILE_sum``", "Total number of L1 volatile pixels or buffers from texture addressing unit"
  "``TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum``", "Total number of cycles tag RAM conflict stalls on a write"

Hardware counter over all texture data unit instances
--------------------------------------------------------

.. csv-table::
  :header: "Hardware counter", "Definition"

  "``TD_ATOMIC_WAVEFRONT_sum``", "Total number of atomic wavefront instructions"
  "``TD_COALESCABLE_WAVEFRONT_sum``", "Total number of coalescable wavefronts according to texture addressing unit"
  "``TD_LOAD_WAVEFRONT_sum``", "Total number of wavefront instructions (read, write, atomic)"
  "``TD_SPI_STALL_sum``", "Total number of cycles texture data unit is stalled by shader processor input"
  "``TD_STORE_WAVEFRONT_sum``", "Total number of write wavefront instructions"
  "``TD_TC_STALL_sum``", "Total number of cycles texture data unit is stalled waiting for texture cache data"
  "``TD_TD_BUSY_sum``", "Total number of texture data unit busy cycles while it is processing or waiting for data"
