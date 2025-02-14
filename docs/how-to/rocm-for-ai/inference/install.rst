.. meta::
   :description: How to install ROCm and popular machine learning frameworks.
   :keywords: ROCm, AI, LLM, train, fine-tune, FSDP, DeepSpeed, LLaMA, tutorial

.. _rocm-for-ai-install:

***********************************************
Installing ROCm and machine learning frameworks
***********************************************

Before getting started, install ROCm and supported machine learning frameworks.

.. grid:: 1

   .. grid-item-card:: Pre-install

      Each release of ROCm supports specific hardware and software configurations. Before installing, consult the
      :doc:`System requirements <rocm-install-on-linux:reference/system-requirements>` and
      :doc:`Installation prerequisites <rocm-install-on-linux:install/prerequisites>` guides.

If you’re new to ROCm, refer to the :doc:`ROCm quick start install guide for Linux
<rocm-install-on-linux:install/quick-start>`.

If you’re using a Radeon GPU for graphics-accelerated applications, refer to the
`Radeon installation instructions <https://rocm.docs.amd.com/projects/radeon/en/docs-6.1.3/docs/install/native_linux/install-radeon.html>`_.

ROCm supports multiple :doc:`installation methods <rocm-install-on-linux:install/install-overview>`:

* :doc:`Using your Linux distribution's package manager <rocm-install-on-linux:install/install-methods/package-manager-index>`

* :doc:`Using the AMDGPU installer <rocm-install-on-linux:install/amdgpu-install>`

* :ref:`Multi-version installation <rocm-install-on-linux:installation-types>`.

.. grid:: 1

   .. grid-item-card:: Post-install

      Follow the :doc:`post-installation instructions <rocm-install-on-linux:install/post-install>` to
      configure your system linker, PATH, and verify the installation.

      If you encounter any issues during installation, refer to the
      :doc:`Installation troubleshooting <rocm-install-on-linux:reference/install-faq>` guide.

Machine learning frameworks
===========================

ROCm supports popular machine learning frameworks and libraries including `PyTorch
<https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package>`_, `TensorFlow
<https://tensorflow.org>`_, `JAX <https://jax.readthedocs.io/en/latest>`_, and `DeepSpeed
<https://cloudblogs.microsoft.com/opensource/2022/03/21/supporting-efficient-large-model-training-on-amd-instinct-gpus-with-deepspeed/>`_.

Review the framework installation documentation. For ease-of-use, it's recommended to use official ROCm prebuilt Docker
images with the framework pre-installed.

* :doc:`PyTorch for ROCm <rocm-install-on-linux:install/3rd-party/pytorch-install>`

* :doc:`TensorFlow for ROCm <rocm-install-on-linux:install/3rd-party/tensorflow-install>`

* :doc:`JAX for ROCm <rocm-install-on-linux:install/3rd-party/jax-install>`

The sections that follow in :doc:`Training a model <../training/train-a-model>` are geared for a ROCm with PyTorch installation.
