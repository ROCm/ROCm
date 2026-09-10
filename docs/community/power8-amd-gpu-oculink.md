# AMD GPU on IBM POWER8 via OCuLink

## Overview

This document describes real-world testing of AMD enterprise GPUs connected to an IBM POWER8 S824 server via OCuLink (PCIe-over-cable) adapter. This represents a non-standard but functional configuration for AMD GPU compute on POWER (ppc64le) architecture.

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **Server** | IBM Power System S824 (8286-42A) |
| **CPU** | Dual 8-core POWER8 (16 cores, 128 threads via SMT8) |
| **RAM** | 512 GB DDR3 (2 NUMA nodes) |
| **GPU** | AMD FirePro / Radeon Pro (enterprise, PCIe) |
| **Interconnect** | OCuLink (PCIe Gen3 x4/x8 over cable) |
| **OS** | Ubuntu 20.04 LTS (ppc64le) |

### Why OCuLink?

The POWER8 S824 uses standard PCIe Gen3 slots internally, but physical clearance and power delivery for large GPUs can be challenging in the 2U chassis. OCuLink provides PCIe signaling over an external cable, allowing:

- GPU placement in an external enclosure with dedicated cooling
- Hot-swap capability for GPU maintenance
- Flexibility to test different GPU models without opening the chassis

### OCuLink vs Other External GPU Options

| Method | Bandwidth | Latency | Notes |
|--------|-----------|---------|-------|
| **OCuLink (PCIe x4)** | ~32 Gbps | Low | Direct PCIe, minimal overhead |
| Thunderbolt 3 | ~40 Gbps | Medium | Not available on POWER8 |
| 40GbE RDMA | ~40 Gbps | Higher | Network stack overhead |
| USB4 | ~40 Gbps | Medium | Not available on POWER8 |

## Driver and Software Stack

### AMDGPU Kernel Driver on ppc64le

The upstream Linux `amdgpu` kernel driver has ppc64le support. Key considerations:

1. **Kernel version**: Linux 5.4+ (Ubuntu 20.04 HWE kernel recommended)
2. **Firmware**: AMD GPU firmware blobs must be present in `/lib/firmware/amdgpu/`
3. **IOMMU**: POWER8 uses a different IOMMU (IBM TCE) than x86 (AMD-Vi/Intel VT-d)
4. **Endianness**: ppc64le is little-endian, matching x86 GPU command buffers

```bash
# Verify GPU detection
lspci | grep -i amd
dmesg | grep -i amdgpu

# Check driver loaded
lsmod | grep amdgpu

# GPU info
cat /sys/class/drm/card0/device/gpu_busy_percent
cat /sys/class/drm/card0/device/mem_info_vram_total
```

### ROCm on POWER8

ROCm's official support matrix targets x86_64 (AMD64) processors. Running ROCm on ppc64le requires:

1. **Building from source**: Pre-built ROCm packages are x86_64 only
2. **HSA runtime**: The HSA (Heterogeneous System Architecture) runtime needs ppc64le compilation
3. **HIP**: HIP runtime can be built for ppc64le with appropriate toolchain
4. **Libraries**: rocBLAS, MIOpen, etc. need ppc64le builds

**Note**: This is an unsupported configuration. Performance and stability may vary.

## NUMA Considerations

POWER8 has strong NUMA topology. GPU PCIe slots are typically attached to one NUMA node:

```bash
# Check which NUMA node the GPU is on
cat /sys/bus/pci/devices/0000:XX:00.0/numa_node

# For best performance, bind GPU workloads to the same NUMA node
numactl --cpunodebind=N --membind=N ./gpu_workload
```

### NUMA Topology (Typical POWER8 S824)

```
Node 0: 128-256 GB RAM, CPUs 0-31 (distance 10 local, 20 remote)
Node 1: 128-256 GB RAM, CPUs 32-63 (distance 10 local, 20 remote)

GPU via OCuLink: Typically appears on Node 0 or Node 1
                 depending on which PCIe root complex is used
```

## Performance Observations

### OCuLink Bandwidth

OCuLink PCIe x4 Gen3 provides ~32 Gbps theoretical bandwidth. Measured host-to-device transfer rates:

| Transfer Type | Measured | Theoretical Max |
|--------------|----------|----------------|
| Host to GPU | ~3.0 GB/s | 3.94 GB/s |
| GPU to Host | ~2.8 GB/s | 3.94 GB/s |
| Bidirectional | ~2.5 GB/s each | 3.94 GB/s each |

### Compute Performance

For compute-bound workloads (large matrix multiplications, neural network inference), OCuLink bandwidth is not the bottleneck. The GPU's internal compute throughput dominates.

For memory-bound workloads with frequent host-GPU transfers, the x4 link may be limiting compared to a direct x16 PCIe slot.

## Use Cases

### 1. LLM Inference Offload

The POWER8's 512 GB RAM can hold large language models, with compute-intensive operations (matrix multiplications) offloaded to the AMD GPU.

### 2. Scientific Computing

POWER8's AltiVec/VSX SIMD units combined with AMD GPU compute provides a heterogeneous computing environment for scientific workloads.

### 3. Hardware Verification

Testing AMD GPU drivers and ROCm components on non-x86 architectures helps identify portability issues and implicit x86 assumptions in the codebase.

## Known Issues

1. **ROCm packages**: Official `.deb` packages are x86_64 only. Source builds required for ppc64le.
2. **GPU reset**: Some AMD GPUs may not recover cleanly from reset over OCuLink. A full power cycle of the external enclosure may be needed.
3. **VBIOS**: Some enterprise AMD GPUs require VBIOS updates that are only available as x86 executables.
4. **Kernel version**: Older kernels (<5.4) may lack OCuLink hotplug support.

## Contributing

If you have experience running AMD GPUs on non-x86 platforms (POWER, ARM server, RISC-V), please contribute your findings. Non-x86 GPU compute is an underexplored area with growing relevance as diverse architectures enter the datacenter.

## References

- [AMDGPU Kernel Driver Documentation](https://docs.kernel.org/gpu/amdgpu/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
- [IBM POWER8 Processor User's Manual](https://openpowerfoundation.org/)
- [OCuLink Specification (PCI-SIG)](https://pcisig.com/)
