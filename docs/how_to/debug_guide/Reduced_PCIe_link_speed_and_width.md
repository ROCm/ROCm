#  **Reduced PCIe link speed and width**

**Error signature:**

NOTE: Following commands and debug steps demonstrated in this section are on a system containing MI250 GPUs. 
Device IDs for different Instinct MI-series GPU are as below:

| GPU    | Device ID |
|--------|-----------|
| MI200  | 740c      |
| MI100  | 738c      |

-   Link speed should be 16GT/s and Width should be x16 for all MI250 devices. The values below this are an indication of sub-optimal PCIe link training and will likely result in reduced performance.

    In the example below, the device has trained at PCIe Gen3 speed (8GT/s).
    
    ```
    $ sudo lspci -d 1002:740c -vvv |grep -e "LnkSta\|Display"
    29:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Aldebaran (rev 01)
    LnkSta: Speed 8GT/s (ok), Width x16 (ok)
    ```
    
-   In configurations where there are intermediate PCIe switches/Bridges between the CPU root port and the MI250 device, the intermediate links may fail to train at the optimal speed and width. If this occurs, then reduced performance will likely be observed.

    To determine if upstream PCIe links to the OAM have sub-optimal PCIe link training, commands like `lspci -vt` and `lshw` can be used to determine the intermediate PCIe links between the CPU root port and MI250 device. Those individual bridge/switch ports can be queried using the command below. In this example, the BIOS configuration was  modified and forced to a lower link speed.

    ```
    $ sudo lspci -vv -s 18:03.1 | grep GT
    LnkCap: Port #0, Speed 5GT/s, Width x16, ASPM L1, Exit Latency L1 <64us
    LnkSta: Speed 5GT/s (ok), Width x16 (ok)
    LnkCap2: Supported Link Speeds: 2.5-5GT/s, Crosslink- Retimer- 2Retimers- DRS-
    LnkCtl2: Target Link Speed: 5GT/s, EnterCompliance- SpeedDis-
    Capabilities: [410 v1] Physical Layer 16.0 GT/s <?>
    ```
 
**Steps to debug:**

1.  Confirm the System BIOS configuration settings for the PCIe ports associated with the MI250 OAM modules.

    System BIOS configuration options are typically available that enable artificially restricting or modifying the maximum PCIe width and speed to utilize during PCIe link training at system boot time. By default, the BIOS configuration should allow PCIe link training at the maximum allowable values; however, inadvertent configuration changes may occur. Refer to the server manufacturer documentation for details on related BIOS configuration options.

2.  Confirm that the server is running the latest supported BIOS and
    platform firmware.

3.  If the previous steps have not addressed the issue, then this may be
    an indication of faulty hardware. Refer to the section, Common
    Hardware Debug Methods, for additional details.

