# **ROCm Installation Issues** 

Please contact AMD technical support <instinct-support@amd.com>, if you see failures even after meeting the below criteria:
1. Met the [prerequisites](https://rocm.docs.amd.com/en/latest/deploy/linux/prerequisites.html)
2. Installed on supported OS and
3. Followed [installation steps](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) as recommended.

For docker based deployments including base images and application
images, refer [here](https://rocm.docs.amd.com/en/latest/deploy/docker.html).

**Common issues**

-   Most of the installation problems are usually due to stale repository links and this arise if there is a old repository setup on previously used system either for ROCm or some other repository. To reset system to a clean state, verify following:

    -   Under `/etc/apt/sources` (or `/etc/yum/sources.d` for yum based
        installation or `/etc/zypp/repos.d` for SLES), look for any
        repository file that has a name rocm.repo, amdgpu.repo and
        delete or disable (or move them to other location as needed) other
        invalid repositories if necessary. 

-   Some installation may require installing standard linux tools, commands
    and packages.

    -   gpgkey related instruction may require to install gpg
        utility i.e, `apt/yum/zypper install gpg --y`.

    -   if amdgpu-install script is used, it relies on dkms,
        therefore verify dkms is installed, `apt/yum/zypper install
        dkms --y`

    -   To see whether particular linux packages/tools needed to be
        installed, look closely to error messages for any particular
        installation step that failed.

-   In rare cases, the installation may fail when a particular Linux
    installation package state or dependency is corrupt preventing
    packages from being installed. Troubleshooting, linux package
    management is outside the scope of this guide. Depending on state of
    the package manager, additional package manager installation options
    may be necessary to be applied (to do a purge, clean, delete older
    version, clean the package manager state etc)

    -   Sometimes packages will fail requiring issuing commands to fix
        the broken state, examples being:

        -   `apt --fix-broken` install before installing package(s).

        -   `--skip-broken` to skip over broken packages and install the
            rest

    -   Detailed technical information about internals of apt/yum and
        zypper are outside the scope of this documentation, instead can
        be found in following:

        -   <https://manpages.ubuntu.com/manpages/xenial/man8/apt.8.html>
            (Ubuntu)

        -   <https://linux.die.net/man/8/yum> (RHEL, Centos Stream)

        -   <https://en.opensuse.org/SDB:Zypper_manual_(plain)> (SLES)
