# Traditional first make target
all:

# Use bash as a shell
# On Ubuntu sh is 'dash'
SHELL:=bash

# Allow RELEASE_FLAG to be overwritten
RELEASE_FLAG?=-r

# Set SANITIZER_FLAG for sanitizer
ASAN_DEP:=
ifeq (${ENABLE_ADDRESS_SANITIZER},true)
	ASAN_DEP=lightning
	SANITIZER_FLAG=-a
endif

export INFRA_REPO:=ROCm/tools/rocm-build

OUT_DIR:=$(shell . ${INFRA_REPO}/envsetup.sh >/dev/null 2>&1 ; echo $${OUT_DIR})
ROCM_INSTALL_PATH:=$(shell . ${INFRA_REPO}/envsetup.sh >/dev/null 2>&1 ; echo $${ROCM_INSTALL_PATH})
PKGTYPE:=$(shell . ${INFRA_REPO}/envsetup.sh >/dev/null 2>&1 ; echo $${PKGTYPE})
DISTRO_NAME:=$(shell . ${INFRA_REPO}/envsetup.sh >/dev/null 2>&1 ; echo $${DISTRO_NAME})

ifeq (${PKGTYPE},)
    # PKGTYPE was not set correctly (probably unsupported distro), error out before scripts fail downstream
    $(error No package type is defined in envsetup.sh for distro name "$(DISTRO_NAME)")
endif

$(info OUT_DIR=${OUT_DIR})
$(info ROCM_INSTALL_PATH=${ROCM_INSTALL_PATH})

# -------------------------------------------------------------------------
# Internal stuff. Could be put in a different file to hide it.
# Internal macros, they need to be defined before being used.

# The internal "eval" allows parts of the Makefile to be generated.
# Whilst it is possible to dump the effective Makefile, it can be
# hard to see where parts come from. Set up the "peval" macro which
# optionally prints out the generated makefile snippet and evaluate it.
# Use "make PEVAL=1 all" to see the things being evaluated.
ifeq (,${PEVAL})
    define peval =
    $(eval $1)
    endef
else
    define peval =
    $(eval $(info $1)$1)
    endef
endif

# macro to add dependencies. Saves having to put all the OUT_DIR/logs in
# The outer strip is to work around a gnu make 4.1 and earlier bug
# It should not be needed.
define adddep =
$(strip $(call peval,components+= $(1) $(2))
$(foreach comp,$(strip $2),$(call peval,${OUT_DIR}/logs/${1}: ${OUT_DIR}/logs/${comp}))
)
endef
# End of internal stuff that is needed at the start of the file
# -------------------------------------------------------------------------

# Dependencies. These can be updated. Anything that is mentioned in
# either the args to the adddep macro will be added to components.  as
# an example there is no need for the adddep of lightning, as it
# depends on nothing and at least one other component includes it.

# Syntax. Up to the first comma everything is fixed. The "call" is a
# keyword to gnu make. The "adddep" is the name of the variable containing
# the macro.
# The second comma delimited argument is the target.
# The third comma delimited arg is the thing that the target depends on.
# It is a space seperated list with zero or more elements.

$(call adddep,amd_smi_lib,${ASAN_DEP})
$(call adddep,aqlprofile,${ASAN_DEP} rocr)
$(call adddep,comgr,lightning devicelibs)
$(call adddep,dbgapi,rocr comgr)
$(call adddep,devicelibs,lightning)
$(call adddep,hip_on_rocclr,${ASAN_DEP} rocr comgr hipcc rocprofiler-register)
$(call adddep,hipcc,)
$(call adddep,hipify_clang,hip_on_rocclr lightning)
$(call adddep,lightning,)
$(call adddep,opencl_on_rocclr,${ASAN_DEP} rocr comgr)
$(call adddep,openmp_extras,lightning devicelibs rocr)
$(call adddep,rocm_bandwidth_test,${ASAN_DEP} rocr)
$(call adddep,rocm_smi_lib,${ASAN_DEP})
$(call adddep,rocm-cmake,${ASAN_DEP})
$(call adddep,rocm-core,${ASAN_DEP})
$(call adddep,rocm-gdb,dbgapi)
$(call adddep,rocminfo,${ASAN_DEP} rocr)
$(call adddep,rocprofiler-register,${ASAN_DEP})
$(call adddep,rocprofiler-sdk,${ASAN_DEP} rocr aqlprofile opencl_on_rocclr hip_on_rocclr comgr)
$(call adddep,rocprofiler-systems,${ASAN_DEP} hipcc rocr hip_on_rocclr rocm_smi_lib rocprofiler roctracer rocprofiler-sdk)
$(call adddep,rocprofiler,${ASAN_DEP} rocr roctracer aqlprofile opencl_on_rocclr hip_on_rocclr comgr)
$(call adddep,rocprofiler-compute,${ASAN_DEP})
$(call adddep,rocr,${ASAN_DEP} lightning rocm_smi_lib devicelibs rocprofiler-register)
$(call adddep,rocr_debug_agent,${ASAN_DEP} hip_on_rocclr rocr dbgapi)
$(call adddep,roctracer,${ASAN_DEP} rocr hip_on_rocclr)

# rocm-dev points to all possible last finish components of Stage1 build.
rocm-dev-components :=amd_smi_lib aqlprofile comgr dbgapi devicelibs hip_on_rocclr hipcc hipify_clang \
	lightning rocprofiler-compute opencl_on_rocclr openmp_extras rocm_bandwidth_test rocm_smi_lib \
	rocm-cmake rocm-core rocm-gdb rocminfo rocprofiler-register rocprofiler-sdk rocprofiler-systems \
	rocprofiler rocr rocr_debug_agent roctracer
$(call adddep,rocm-dev,$(filter-out ${NOBUILD} kernel_ubuntu,${rocm-dev-components}))

$(call adddep,amdmigraphx,hip_on_rocclr half rocblas miopen-hip lightning hipcc hiptensor)
$(call adddep,composable_kernel,lightning hipcc hip_on_rocclr rocm-cmake)
$(call adddep,half,rocm-cmake)
$(call adddep,hipblas-common,lightning)
$(call adddep,hipblas,hip_on_rocclr rocblas rocsolver lightning hipcc)
$(call adddep,hipblaslt,hip_on_rocclr openmp_extras lightning hipcc hipblas-common rocm-dev)
$(call adddep,hipcub,hip_on_rocclr rocprim lightning hipcc)
$(call adddep,hipfft,hip_on_rocclr openmp_extras rocfft rocrand hiprand lightning hipcc)
$(call adddep,hipfort,rocblas hipblas rocsparse hipsparse rocfft hipfft rocrand hiprand rocsolver hipsolver lightning hipcc)
$(call adddep,hiprand,hip_on_rocclr rocrand lightning hipcc)
$(call adddep,hipsolver,hip_on_rocclr rocblas rocsolver rocsparse lightning hipcc hipsparse)
$(call adddep,hipsparse,hip_on_rocclr rocsparse lightning hipcc)
$(call adddep,hipsparselt,hip_on_rocclr hipsparse lightning hipcc openmp_extras)
$(call adddep,hiptensor,hip_on_rocclr composable_kernel lightning hipcc)
$(call adddep,miopen-deps,lightning hipcc)
$(call adddep,miopen-hip,composable_kernel half hip_on_rocclr miopen-deps hipblas hipblaslt rocrand roctracer lightning hipcc)
$(call adddep,mivisionx,amdmigraphx miopen-hip rpp lightning hipcc)
$(call adddep,rccl,rocm-core hip_on_rocclr rocr lightning hipcc rocm_smi_lib hipify_clang)
$(call adddep,rdc,rocm_smi_lib rocprofiler rocmvalidationsuite)
$(call adddep,rocalution,rocblas rocsparse rocrand lightning hipcc)
$(call adddep,rocblas,hip_on_rocclr openmp_extras lightning hipcc hipblaslt)
$(call adddep,rocal,mivisionx)
$(call adddep,rocdecode,hip_on_rocclr lightning hipcc amdmigraphx)
$(call adddep,rocfft,hip_on_rocclr rocrand hiprand lightning hipcc openmp_extras)
$(call adddep,rocjpeg,hip_on_rocclr lightning hipcc rocm-dev)
$(call adddep,rocmvalidationsuite,hip_on_rocclr rocr hipblas hiprand hipblaslt rocm-core lightning hipcc rocm_smi_lib)
$(call adddep,rocprim,hip_on_rocclr lightning hipcc)
$(call adddep,rocrand,hip_on_rocclr lightning hipcc)
$(call adddep,rocsolver,hip_on_rocclr rocblas rocsparse rocprim lightning hipcc)
$(call adddep,rocsparse,hip_on_rocclr rocprim lightning hipcc)
$(call adddep,rocthrust,hip_on_rocclr rocprim lightning hipcc)
$(call adddep,rocwmma,hip_on_rocclr rocblas lightning hipcc rocm-cmake rocm_smi_lib)
$(call adddep,rpp,half lightning hipcc openmp_extras)
$(call adddep,transferbench,hip_on_rocclr lightning hipcc)


# -------------------------------------------------------------------------
# The rest of the file is internal
# Do not pass jobserver params if -n build
ifneq (,$(findstring n,${MAKEFLAGS}))
RMAKE:=
else
RMAKE := +
endif


# disable the builtin rules
.SUFFIXES:

# Linear
# include moredeps

# A macro to define a toplevel target, add it to the 'all' target
# Make it depend on the generated log. Generate the log of the build.

# See if the macro is already defined, if so don't touch it.
# As GNU make allows more than one makefile to be specified with "-f"
# one could put an alternative definition of "toplevel" in a different
# file or even the environment, and use the data in this file for other
# purposes. Uses might include generating output in "dot" format for
# showing the dependency graph, or having a wrapper script to run programs
# to generate code quality tools.
ifeq (${toplevel},)
# { Start of test to see if toplevel is defined
define toplevel =

# The "target" make, this builds the package if it is out of date
T_$1: ${OUT_DIR}/logs/$1 FRC
	:              $1 built

# The "upload" for $1, it uploads the packages for $1 to the central storage
U_$1: T_$1 FRC
	source $${INFRA_REPO}/envsetup.sh && $${INFRA_REPO}/upload_packages.sh "$1"
	:		$1 uploaded

# The "clean" for $1, it just marks the target as not existing so it will be built
# in the future.
C_$1: FRC
	rm -f ${OUT_DIR}/logs/$1 ${OUT_DIR}/logs/$1.repackaged

# parallel build {
${OUT_DIR}/logs/$1: | ${OUT_DIR}/logs
ifneq ($(wildcard ${OUT_DIR}/logs/$1.repackaged),)
	@echo  Skipping build of $1 as it has already been repackaged
	cat $$@.repackaged > $$@
	rm -f $$@.repackaged
else # } {
	@echo  $1 started due to $$? | sed "s:${OUT_DIR}/logs/::g"
# Build in a subshell so we get the time output
# Pass in jobserver info using the RMAKE variable
	${RMAKE}@( if set -x && source $${INFRA_REPO}/envsetup.sh && \
	rm -f $$@.errors $$@ $$@.repackaged && \
	$${INFRA_REPO}/build_$1.sh -c && \
	time bash -x $${INFRA_REPO}/build_$1.sh $${RELEASE_FLAG} $${SANITIZER_FLAG} && $${INFRA_REPO}/post_inst_pkg.sh "$1" ; \
	then mv $$@.inprogress $$@ ; \
	else mv $$@.inprogress $$@.errors ; echo Error in $1 >&2 ; exit 1 ;\
	fi ) > $$@.inprogress 2>&1
endif # }

# end of toplevel macro
endef
# } End of test to see if toplevel is defined
endif

components:=$(sort $(components))

# Create all the T_xxxx and C_xxxx targets
$(call peval,$(foreach dep,$(strip ${components}),$(call toplevel,${dep})))

# Add all the T_xxxx targets to "all"  except those listed in NOBUILD
# Note this does not prohibit them from being built, it just means that
# a build of "all" will not force them to be built directly
# example command
#  make -f jenkins-utils/scripts/Stage1.mk -j60

##help all: Build everything
all: $(addprefix T_,$(filter-out ${NOBUILD},${components}))
	@echo All ROCm components built
# Do not document this target
upload: $(addprefix U_,$(filter-out ${NOBUILD},${components}))
	@echo All ROCm components built and uploaded

upload-rocm-dev: $(addprefix U_,$(filter-out ${NOBUILD},${components}))
	@echo All rocm-dev components built and uploaded

##help rocm-dev: Build a subset of ROCm
rocm-dev: $(addprefix T_,$(filter-out ${NOBUILD},${components}))
	@echo rocm-dev built

${OUT_DIR}/logs:
	sudo mkdir -p -m 775 "${ROCM_INSTALL_PATH}" && \
	sudo chown -R "$(shell id -u):$(shell id -g)" "/opt"
	sudo chown -R "$(shell id -u):$(shell id -g)" "/home/$(shell id -un)"
	mkdir -p "${@}"
	mkdir -p ${HOME}/.ccache

##help clean: remove the output directory and recreate it
clean:
	[ -n "${OUT_DIR}" ] && rm -rf "${OUT_DIR}"
	mkdir -p ${OUT_DIR}/logs

.SECONDARY: ${components:%=${OUT_DIR}/logs/%}

.PHONY: all clean repack help list_components

##help list_components: output the list of components
##help : Hint make list_components | paste - - - | column -t
list_components:
	@echo "${components}" | sed 'y/ /\n/'

##help help:	show this text
help:
	@sed -n 's/^##help //p' ${MAKEFILE_LIST} | \
	    if type -t column > /dev/null ; then  column -s: -t ; else cat ; fi

FRC:
