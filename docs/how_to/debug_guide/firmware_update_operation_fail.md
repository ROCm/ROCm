# **Firmware Update Operation Fails**

**Error signature:**

-   The MI250 firmware operation fails to complete successfully and the device is no longer visible on the PCIe bus following the IFWI update.

**Steps to debug:**

-   Reboot the server and verify that all MI250 devices are present in `lspci` output

-   Ensure that the appropriate SR-IOV settings are configured. Either of these options are acceptable:

    -   Enable SR-IOV in the system BIOS

    -   If SR-IOV is disabled in the system BIOS, then the kernel grub configuration must be updated.

        -   Add `pci=realloc=off` to `GRUB_CMDLINE_LINUX` and update the grub configuration.

        -   Refer to OS documentation on the location of the grub configuration file and method to update grub after making that modification.
