.. meta::
  :description: Supported data types of AMD GPUs and libraries in ROCm.
  :keywords: precision, data types, HIP types, int8, float8, float8 (E4M3),
             float8 (E5M2), bfloat8, float16, half, bfloat16, tensorfloat32,
             float, float32, float64, double, AMD data types, HIP data types,
             ROCm precision, ROCm data types

*************************************************************
Data types and precision support
*************************************************************

This topic lists the data types support on AMD GPUs, ROCm libraries along
with corresponding :doc:`HIP <hip:index>` data types.

Integral types
==============

The signed and unsigned integral types supported by ROCm are listed in
the following table.

.. list-table::
    :header-rows: 1
    :widths: 15,35,50

    *
      - Type name
      - HIP type
      - Description
    *
      - int8
      - ``int8_t``, ``uint8_t``
      - A signed or unsigned 8-bit integer
    *
      - int16
      - ``int16_t``, ``uint16_t``
      - A signed or unsigned 16-bit integer
    *
      - int32
      - ``int32_t``, ``uint32_t``
      - A signed or unsigned 32-bit integer
    *
      - int64
      - ``int64_t``, ``uint64_t``
      - A signed or unsigned 64-bit integer

.. _precision_support_floating_point_types:

Floating-point types
====================

The floating-point types supported by ROCm are listed in the following table.

.. image:: ../data/about/compatibility/floating-point-data-types.png
    :alt: Supported floating-point types

.. list-table::
    :header-rows: 1
    :widths: 15,15,70

    *
      - Type name
      - HIP type
      - Description
    *
      - float8 (E4M3)
      - ``__hip_fp8_e4m3_fnuz``
      - An 8-bit floating-point number that mostly follows IEEE-754 conventions
        and **S1E4M3** bit layout, as described in `8-bit Numerical Formats for Deep Neural Networks <https://arxiv.org/abs/2206.02915>`_,
        with expanded range and no infinity or signed zero. NaN is represented
        as negative zero.
    *
      - float8 (E5M2)
      - ``__hip_fp8_e5m2_fnuz``
      - An 8-bit floating-point number mostly following IEEE-754 conventions and
        **S1E5M2** bit layout, as described in `8-bit Numerical Formats for Deep Neural Networks <https://arxiv.org/abs/2206.02915>`_,
        with expanded range and no infinity or signed zero. NaN is represented
        as negative zero.
    *
      - float16
      - ``half``
      - A 16-bit floating-point number that conforms to the IEEE 754-2008
        half-precision storage format.
    *
      - bfloat16
      - ``bfloat16``
      - A shortened 16-bit version of the IEEE 754 single-precision storage
        format.
    *
      - tensorfloat32
      - Not available
      - A floating-point number that occupies 32 bits or less of storage,
        providing improved range compared to half (16-bit) format, at
        (potentially) greater throughput than single-precision (32-bit) formats.
    *
      - float32
      - ``float``
      - A 32-bit floating-point number that conforms to the IEEE 754
        single-precision storage format.
    *
      - float64
      - ``double``
      - A 64-bit floating-point number that conforms to the IEEE 754
        double-precision storage format.

.. note::

  * The float8 and tensorfloat32 types are internal types used in calculations
    in Matrix Cores and can be stored in any type of the same size.

  * The encodings for FP8 (E5M2) and FP8 (E4M3) that the
    MI300 series natively supports differ from the FP8 (E5M2) and FP8 (E4M3)
    encodings used in NVIDIA H100
    (`FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_).

  * In some AMD documents and articles, float8 (E5M2) is referred to as bfloat8.

  * The :doc:`low precision floating point types page <hip:reference/low_fp_types>`
    describes how to use these types in HIP with examples.

Level of support definitions
============================

In the following sections, icons represent the level of support. These icons,
described in the following table, are also used in the library data type support
pages.

.. list-table::
    :header-rows: 1

    *
      -  Icon
      - Definition

    *
      - NA
      - Not applicable

    *
      - ❌
      - Not supported

    *
      - ⚠️
      - Partial support

    *
      - ✅
      - Full support

.. note::

  * Full support means that the type is supported natively or with hardware
    emulation.

  * Native support means that the operations for that type are implemented in
    hardware. Types that are not natively supported are emulated with the
    available hardware. The performance of non-natively supported types can
    differ from the full instruction throughput rate. For example, 16-bit
    integer operations can be performed on the 32-bit integer ALUs at full rate;
    however, 64-bit integer operations might need several instructions on the
    32-bit integer ALUs.

  * Any type can be emulated by software, but this page does not cover such
    cases.

Data type support by Hardware Architecture
==========================================

The MI200 series GPUs, which include MI210, MI250, and MI250X, are based on the
CDNA2 architecture. The MI300 series GPUs, consisting of MI300A, MI300X, and
MI325X, are based on the CDNA3 architecture.

Compute units support
---------------------

The following table lists data type support for compute units.

.. tab-set::

  .. tab-item:: Integral types
    :sync: integral-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - int8
        - int16
        - int32
        - int64
      *
        - MI100
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - MI200 series
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - MI300 series
        - ✅
        - ✅
        - ✅
        - ✅

  .. tab-item:: Floating-point types
    :sync: floating-point-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - float8 (E4M3)
        - float8 (E5M2)
        - float16
        - bfloat16
        - tensorfloat32
        - float32
        - float64
      *
        - MI100
        - ❌
        - ❌
        - ✅
        - ✅
        - ❌
        - ✅
        - ✅
      *
        - MI200 series
        - ❌
        - ❌
        - ✅
        - ✅
        - ❌
        - ✅
        - ✅
      *
        - MI300 series
        - ❌
        - ❌
        - ✅
        - ✅
        - ❌
        - ✅
        - ✅

Matrix core support
-------------------

The following table lists data type support for AMD GPU matrix cores.

.. tab-set::

  .. tab-item:: Integral types
    :sync: integral-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - int8
        - int16
        - int32
        - int64
      *
        - MI100
        - ✅
        - ❌
        - ❌
        - ❌
      *
        - MI200 series
        - ✅
        - ❌
        - ❌
        - ❌
      *
        - MI300 series
        - ✅
        - ❌
        - ❌
        - ❌

  .. tab-item:: Floating-point types
    :sync: floating-point-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - float8 (E4M3)
        - float8 (E5M2)
        - float16
        - bfloat16
        - tensorfloat32
        - float32
        - float64
      *
        - MI100
        - ❌
        - ❌
        - ✅
        - ✅
        - ❌
        - ✅
        - ❌
      *
        - MI200 series
        - ❌
        - ❌
        - ✅
        - ✅
        - ❌
        - ✅
        - ✅
      *
        - MI300 series
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅

Atomic operations support
-------------------------

The following table lists data type support for atomic operations.

.. tab-set::

  .. tab-item:: Integral types
    :sync: integral-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - int8
        - int16
        - int32
        - int64
      *
        - MI100
        - ❌
        - ❌
        - ✅
        - ❌
      *
        - MI200 series
        - ❌
        - ❌
        - ✅
        - ✅
      *
        - MI300 series
        - ❌
        - ❌
        - ✅
        - ✅

  .. tab-item:: Floating-point types
    :sync: floating-point-type

    .. list-table::
      :header-rows: 1

      *
        - Type name
        - float8 (E4M3)
        - float8 (E5M2)
        - float16
        - bfloat16
        - tensorfloat32
        - float32
        - float64
      *
        - MI100
        - ❌
        - ❌
        - ✅
        - ❌
        - ❌
        - ✅
        - ❌
      *
        - MI200 series
        - ❌
        - ❌
        - ✅
        - ❌
        - ❌
        - ✅
        - ✅
      *
        - MI300 series
        - ❌
        - ❌
        - ✅
        - ❌
        - ❌
        - ✅
        - ✅

.. note::

  You can emulate atomic operations using software for cases that are not
  natively supported. Software-emulated atomic operations have a high negative
  performance impact when they frequently access the same memory address.

Data type support in ROCm libraries
===================================

ROCm library support for int8, float8 (E4M3), float8 (E5M2), int16, float16,
bfloat16, int32, tensorfloat32, float32, int64, and float64 is listed in the
following tables.

Libraries input/output type support
-----------------------------------

The following tables list ROCm library support for specific input and output
data types. Refer to the corresponding library data type support page for a
detailed description.

.. tab-set::

  .. tab-item:: Integral types
    :sync: integral-type

    .. list-table::
      :header-rows: 1

      *
        - Library input/output data type name
        - int8
        - int16
        - int32
        - int64
      *
        - :doc:`hipSPARSELt <hipsparselt:reference/data-type-support>`
        - ✅/✅
        - ❌/❌
        - ❌/❌
        - ❌/❌
      *
        - :doc:`rocRAND <rocrand:api-reference/data-type-support>`
        - NA/✅
        - NA/✅
        - NA/✅
        - NA/✅
      *
        - :doc:`hipRAND <hiprand:api-reference/data-type-support>`
        - NA/✅
        - NA/✅
        - NA/✅
        - NA/✅
      *
        -  :doc:`rocPRIM <rocprim:reference/data-type-support>`
        - ✅/✅
        - ✅/✅
        - ✅/✅
        - ✅/✅
      *
        - :doc:`hipCUB <hipcub:api-reference/data-type-support>`
        - ✅/✅
        - ✅/✅
        - ✅/✅
        - ✅/✅
      *
        - :doc:`rocThrust <rocthrust:data-type-support>`
        - ✅/✅
        - ✅/✅
        - ✅/✅
        - ✅/✅

  .. tab-item:: Floating-point types
    :sync: floating-point-type

    .. list-table::
      :header-rows: 1

      *
        - Library input/output data type name
        - float8 (E4M3)
        - float8 (E5M2)
        - float16
        - bfloat16
        - tensorfloat32
        - float32
        - float64
      *
        - :doc:`hipSPARSELt <hipsparselt:reference/data-type-support>`
        - ❌/❌
        - ❌/❌
        - ✅/✅
        - ✅/✅
        - ❌/❌
        - ❌/❌
        - ❌/❌
      *
        - :doc:`rocRAND <rocrand:api-reference/data-type-support>`
        - NA/❌
        - NA/❌
        - NA/✅
        - NA/❌
        - NA/❌
        - NA/✅
        - NA/✅
      *
        - :doc:`hipRAND <hiprand:api-reference/data-type-support>`
        - NA/❌
        - NA/❌
        - NA/✅
        - NA/❌
        - NA/❌
        - NA/✅
        - NA/✅
      *
        - :doc:`rocPRIM <rocprim:reference/data-type-support>`
        - ❌/❌
        - ❌/❌
        - ✅/✅
        - ✅/✅
        - ❌/❌
        - ✅/✅
        - ✅/✅
      *
        - :doc:`hipCUB <hipcub:api-reference/data-type-support>`
        - ❌/❌
        - ❌/❌
        - ✅/✅
        - ✅/✅
        - ❌/❌
        - ✅/✅
        - ✅/✅
      *
        - :doc:`rocThrust <rocthrust:data-type-support>`
        - ❌/❌
        - ❌/❌
        - ⚠️/⚠️
        - ⚠️/⚠️
        - ❌/❌
        - ✅/✅
        - ✅/✅

.. note::
   
  As random number generation libraries, rocRAND and hipRAND only specify output
  data types for the random values they generate, with no need for input data
  types.

Libraries internal calculations type support
--------------------------------------------

The following tables list ROCm library support for specific internal data types.
Refer to the corresponding library data type support page for a detailed
description.

.. tab-set::

  .. tab-item:: Integral types
    :sync: integral-type

    .. list-table::
      :header-rows: 1

      *
        - Library internal data type name
        - int8
        - int16
        - int32
        - int64
      *
        - :doc:`hipSPARSELt <hipsparselt:reference/data-type-support>`
        - ❌
        - ❌
        - ✅
        - ❌


  .. tab-item:: Floating-point types
    :sync: floating-point-type

    .. list-table::
      :header-rows: 1

      *
        - Library internal data type name
        - float8 (E4M3)
        - float8 (E5M2)
        - float16
        - bfloat16
        - tensorfloat32
        - float32
        - float64
      *
        - :doc:`hipSPARSELt <hipsparselt:reference/data-type-support>`
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
        - ✅
        - ❌
