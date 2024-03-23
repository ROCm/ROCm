# **System Crashes when using a virtual function (VF) under SRIOV virtualization**

**Error signature:**

-   System crash / NMI

-   IO PAGE FAULT and IRQ remapping does not support X2APIC mode error message in dmesg logs

**Steps to debug:**

-   IOMMU must be disabled or set to passthrough in the grub configuration

    -   Verify that IOMMU is disabled in the system BIOS

    -   If IOMMU is enabled in the system BIOS, then the kernel grub configuration must be updated.

        -   For systems with an AMD-based processor, add `amd_iommu=on iommu=pt` to GRUB_CMDLINE_LINUX and update the grub configuration.

        -   For systems with an Intel-based processor, add `intel_iommu=on iommu=pt` to GRUB_CMDLINE_LINUX and update the grub configuration.

        -   Refer to OS documentation on the location of the grub configuration file and method to update grub after making that modification.

-   Reboot the system following the grub configuration changes.

