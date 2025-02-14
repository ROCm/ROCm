---
myst:
  html_meta:
    "description": "Learn more about common system-level debugging measures for ROCm."
    "keywords": "env, var, sys, PCIe, troubleshooting, admin, error"
---

# System debugging

## ROCm language and system-level debug, flags, and environment variables

Kernel options to avoid: the Ethernet port getting renamed every time you change graphics cards, `net.ifnames=0 biosdevname=0`

## ROCr error code

* 2 Invalid Dimension
* 4 Invalid Group Memory
* 8 Invalid (or Null) Code
* 32 Invalid Format
* 64 Group is too large
* 128 Out of VGPRs
* 0x80000000 Debug Options

## Command to dump firmware version and get Linux kernel version

`sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info`

`uname -a`

## Debug flags

Debug messages when developing/debugging base ROCm driver. You could enable the printing from `libhsakmt.so` by setting an environment variable, `HSAKMT_DEBUG_LEVEL`. Available debug levels are 3-7. The higher level you set, the more messages will print.

* `export HSAKMT_DEBUG_LEVEL=3` : Only pr_err() prints.

* `export HSAKMT_DEBUG_LEVEL=4` : pr_err() and pr_warn() print.

* `export HSAKMT_DEBUG_LEVEL=5` : We currently do not implement “notice”. Setting to 5 is same as setting to 4.

* `export HSAKMT_DEBUG_LEVEL=6` : pr_err(), pr_warn(), and pr_info print.

* `export HSAKMT_DEBUG_LEVEL=7` : Everything including pr_debug prints.

## ROCr level environment variables for debug

`HSA_ENABLE_SDMA=0`

`HSA_ENABLE_INTERRUPT=0`

`HSA_SVM_GUARD_PAGES=0`

`HSA_DISABLE_CACHE=1`

## Turn off page retry on GFX9/Vega devices

`sudo -s`

`echo 1 > /sys/module/amdkfd/parameters/noretry`

## HIP environment variables 3.x

### OpenCL debug flags

`AMD_OCL_WAIT_COMMAND=1 (0 = OFF, 1 = On)`

## PCIe-debug

For information on how to debug and profile HIP applications, see {doc}`hip:how-to/debugging`
