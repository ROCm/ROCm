<head>
  <meta charset="UTF-8">
  <meta name="description" content="GPU isolation techniques">
  <meta name="keywords" content="GPU isolation techniques, UUID, universally unique identifier,
  environment variables, virtual machines, AMD, ROCm">
</head>

# GPU isolation techniques

Restricting the access of applications to a subset of GPUs, aka isolating
GPUs allows users to hide GPU resources from programs. The programs by default
will only use the "exposed" GPUs ignoring other (hidden) GPUs in the system.

There are multiple ways to achieve isolation of GPUs in the ROCm software stack,
differing in which applications they apply to and the security they provide.
This page serves as an overview of the techniques.

## Environment variables

The runtimes in the ROCm software stack read these environment variables to
select the exposed or default device to present to applications using them.

Environment variables shouldn't be used for isolating untrusted applications,
as an application can reset them before initializing the runtime.

### `ROCR_VISIBLE_DEVICES`

A list of device indices or {abbr}`UUID (universally unique identifier)`s
that will be exposed to applications.

Runtime
: ROCm Software Runtime. Applies to all applications using the user mode ROCm
  software stack.

```{code-block} shell
:caption: Example to expose the 1. device and a device based on UUID.
export ROCR_VISIBLE_DEVICES="0,GPU-DEADBEEFDEADBEEF"
```

### `GPU_DEVICE_ORDINAL`

Devices indices exposed to OpenCL and HIP applications.

Runtime
: ROCm Compute Language Runtime (`ROCclr`). Applies to applications and runtimes
  using the `ROCclr` abstraction layer including HIP and OpenCL applications.

```{code-block} shell
:caption: Example to expose the 1. and 3. device in the system.
export GPU_DEVICE_ORDINAL="0,2"
```

(hip_visible_devices)=

### `HIP_VISIBLE_DEVICES`

Device indices exposed to HIP applications.

Runtime: HIP runtime. Applies only to applications using HIP on the AMD platform.

```{code-block} shell
:caption: Example to expose the 1. and 3. devices in the system.
export HIP_VISIBLE_DEVICES="0,2"
```

### `CUDA_VISIBLE_DEVICES`

Provided for CUDA compatibility, has the same effect as `HIP_VISIBLE_DEVICES`
on the AMD platform.

Runtime
: HIP or CUDA Runtime. Applies to HIP applications on the AMD or NVIDIA platform
  and CUDA applications.

### `OMP_DEFAULT_DEVICE`

Default device used for OpenMP target offloading.

Runtime
: OpenMP Runtime. Applies only to applications using OpenMP offloading.

```{code-block} shell
:caption: Example on setting the default device to the third device.
export OMP_DEFAULT_DEVICE="2"
```

## Docker

Docker uses Linux kernel namespaces to provide isolated environments for
applications. This isolation applies to most devices by default, including
GPUs. To access them in containers explicit access must be granted, please see
{ref}`docker-access-gpus-in-container` for details.
Specifically refer to {ref}`docker-restrict-gpus` on exposing just a subset
of all GPUs.

Docker isolation is more secure than environment variables, and applies
to all programs that use the `amdgpu` kernel module interfaces.
Even programs that don't use the ROCm runtime, like graphics applications
using OpenGL or Vulkan, can only access the GPUs exposed to the container.

## GPU passthrough to virtual machines

Virtual machines achieve the highest level of isolation, because even the kernel
of the virtual machine is isolated from the host. Devices physically installed
in the host system can be passed to the virtual machine using PCIe passthrough.
This allows for using the GPU with a different operating systems like a Windows
guest from a Linux host.

Setting up PCIe passthrough is specific to the hypervisor used. ROCm officially
supports [VMware ESXi](https://www.vmware.com/products/esxi-and-esx.html)
for select GPUs.

<!--
TODO: This should link to a page about virtualization that explains
      pass-through and SR-IOV and how-tos for maybe `libvirt` and `VMWare`
-->
