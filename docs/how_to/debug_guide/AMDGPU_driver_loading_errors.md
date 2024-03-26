# **AMDGPU driver loading errors**

Below lists the dmseg errors related to AMDGPU driver not loading:

-   **GPU init fatal error**
   
    ```
    dmesg | grep -i "Fatal error during GPU init"
    amdgpu 0000:03:00.0: Fatal error during GPU init
    ```
-   **UCX link status**

    ```
    dmesg | grep mlx
    mlx5_core 0000:21:00.0 enp33s0f0: Link up
    mlx5_core 0000:21:00.1 enp33s0f1: Link down
    ```
-   **GPU not visible**

    ```
    kernel: amdgpu 0000:c9:00.0: Direct firmware load for amdgpu/arcturus_gpu_info.bin failed with error -2 
    kernel: amdgpu 0000:c9:00.0: amdgpu: Failed to load gpu_info firmware "amdgpu/arcturus_gpu_info.bin" 
    kernel: amdgpu 0000:c9:00.0: amdgpu: Fatal error during GPU init 
    kernel: amdgpu: probe of 0000:c9:00.0 failed with error -2 
    ``` 
-   **segfault error**

    ```
    segfault at 0 ip 00007f5bcba462e5 sp 00007ffd0810a180 error 4 in libnuma.so.1.0.0[7f5bcba40000+a000]
    ```

-   **doorbell allocation failure**

    ```
    amdgpu: Failed to alloc doorbell for pdd
    amdgpu: Failed to create process device data
    ```

Any of the above errors given will indicate an issue with the KFD Driver not loading properly. One can try the steps stated in [Common hardware debug methods](./Common_hardware_debug_methods.md) to troubleshoot the errors, if not kindly file a ticket on JIRA or send an email to <instinct-support@amd.com> with rocmtechsupport/dmesg logs attached.


