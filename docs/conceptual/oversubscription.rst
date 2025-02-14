.. meta::
   :description: Learn what causes oversubscription.
   :keywords: warning, log, gpu, performance penalty, help

*******************************************************************
Oversubscription of hardware resources in AMD Instinct accelerators
*******************************************************************

When an AMD Instinct™ MI series accelerator enters an oversubscribed state, the ``amdgpu`` driver outputs the following
message.

``amdgpu: Runlist is getting oversubscribed. Expect reduced ROCm performance.``

Oversubscription occurs when application demands exceed the available hardware resources. In an oversubscribed
state, the hardware scheduler tries to manage resource usage in a round-robin fashion. However,
this can result in reduced performance, as resources might be occupied by applications or queues not actively
submitting work. The granularity of hardware resources occupied by an inactive queue can be in the order of
milliseconds, during which the accelerator or GPU is effectively blocked and unable to process work submitted by other
queues.

What triggers oversubscription?
===============================

The system enters an oversubscribed state when one of the following conditions is met:

* **Hardware queue limit exceeded**: The number of user-mode compute queues requested by applications exceeds the
  hardware limit of 24 queues for current Instinct accelerators.

* **Virtual memory context slots exceeded**: The number of user processes exceeds the number of available virtual memory
  context slots, which is 11 for current Instinct accelerators.

* **Multiple processes using cooperative workgroups**: More than one process attempts to use the cooperative workgroup
  feature, leading to resource contention.

