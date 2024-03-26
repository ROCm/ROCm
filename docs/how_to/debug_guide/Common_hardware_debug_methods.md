**Common Hardware Debug Methods** 

It is recommended to attempt non-invasive methods to resolve any observed failures. However, if those are not successful, reseating and/or swapping the OAM modules may resolve the observed failure or provide an indication of a faulty hardware. PCIe device discovery, PCIe link training, and XGMI communication failures are common candidates for these hardware procedures, particularly in cases where these are observed during the initial installation of the hardware.

AMD recommends consulting the server manufacturer's documentation for details on installing or swapping OAM modules. In some cases, the server manufacturer may recommend that these procedures only be performed by authorized support personnel.

**Common Error signatures in Instinct MI250 system:**

-   Subset of MI250 devices not visible in lspci output

    ` $ sudo lspci -d 1002:740c`

-   PCIe link speed and width is below the expected values

    -   Link speed should be 16GT/s and Width should be x16 for all
        MI250 devices.

    ```
    $ sudo lspci -d 1002:740c -vvv |grep -e "LnkSta\|Display"
    29:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Aldebaran (rev 01)
    LnkSta: Speed 16GT/s (ok), Width x16 (ok)
    ```

