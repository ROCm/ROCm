.. meta::
   :description: Discover, run, and benchmark deep learning models with AMD MAD -- Model Automation and Dashboarding tool.
   :keywords: dashboard, machine, deep, container, playground, tune

************************
Running models using MAD
************************

The AMD Model Automation and Dashboarding (MAD) tool integrates an AI model zoo with automated execution capabilities
across various GPU architectures. It facilitates performance tracking by maintaining historical performance data and
generating dashboards for analysis. The MAD source code repository and complete documentation are at
`<https://github.com/ROCm/MAD>`__.

MAD retrieves various models from their repositories and tests their performance in ROCm Docker images. It is an index
of deep learning models optimized for reproducible accuracy and performance on AMD GPUs and accelerators using the ROCm
software stack.

Use MAD to:

*  Try new models

*  Compare performance between patches or architectures

*  Track functionality and performance over time

Getting started with MAD
========================

Refer to the procedures in :doc:`rocm-install-on-linux:index` to set up your host computer with ROCm. Follow the
detailed :doc:`installation instructions <rocm-install-on-linux:install/detailed-install>` for Linux-based platforms.

ROCm Docker images
------------------

ROCm Docker images for PyTorch and TensorFlow are available on Docker Hub at
`rocm/pytorch <https://hub.docker.com/r/rocm/pytorch>`_ and
`rocm/tensorflow <https://hub.docker.com/r/rocm/tensorflow>`_.

A unified Docker image that packages together vLLM and PyTorch for the AMD Instinct™ MI300X accelerator is at
`rocm/vllm <https://hub.docker.com/r/rocm/vllm>`_ . This enables users to quickly validate the expected inference
performance numbers on the MI300X. This Docker image includes:

- ROCm

- vLLM

- PyTorch

- Tuning files (CSV format)

See `<https://github.com/ROCm/MAD/tree/develop/benchmark/vllm>`__ for more information.

.. _mad-run-locally:

Using MAD to run models locally
===============================

The following describes MAD's basic usage and functionalities.

1. Clone the `MAD repository <https://github.com/ROCm/MAD>`_ to a local directory and install the required packages
   on the host machine. For example:

   .. code-block:: shell

      git clone https://github.com/ROCm/MAD
      cd MAD
      pip3 install -r requirements.txt

2. Using the ``tools/run_models.py`` script, you can run and collect performance results for all models in
   ``models.json`` locally on a Docker host. Refer to
   `MAD/models.json <https://github.com/ROCm/blob/develop/models.json>`_.

   ``run_models.py`` is the main MAD command line interface for running models locally. While the tool has many options,
   running any single model is very easy. To run a model, look for its name or tag in the ``models.json`` and pass it to
   ``run_models.py`` in the form of:

   .. code-block:: shell

      tools/run_models.py [-h] [--tags TAGS] [--timeout TIMEOUT] [--live-output] [--clean-docker-cache] [--keep-alive] [--keep-model-dir] [-o OUTPUT] [--log-level LOG-LEVEL]

   See :ref:`mad-run-args` for the list of options and their descriptions.

For each model in ``models.json``, the script:

* Builds Docker images associated with each model. The images are named
  ``ci-$(model_name)``, and are not removed after the script completes.

* Starts the Docker container, with name, ``container_$(model_name)``.
  The container should automatically be stopped and removed whenever
  the script exits.

* Clones the git ``url`` and runs the ``scripts``.

* Compiles the final ``perf.csv`` and ``perf.html``.

.. _mad-run-args:

Arguments
---------

The following list of arguments describe some of the MAD tool's capabilities.

--help, -h
   Show this help message and exit

--tags TAGS
   Tags to run model (can be multiple).

   .. note::

      With the tag functionality, you can select a subset of the models with the corresponding tags to be run. You
      can specify tags with the ``--tags`` argument. If multiple tags are specified, all models that
      match any specified tag are selected.

      Each model name in ``models.json`` is automatically a tag that can be used to run that model. Tags are also supported
      in comma-separated form.

      For example, to run the ``pyt_huggingface_bert`` model, use:

      .. code-block:: shell

         python3 tools/run_models.py --tags pyt_huggingface_bert

      Or, to run all PyTorch models, use:

      .. code-block:: shell

         python3 tools/run_models.py --tags pyt

--timeout TIMEOUT
   Timeout for the application running model in seconds, default timeout of 7200 (2 hours).

--live-output
   Prints output in real-time directly on `STDOUT`.

--clean-docker-cache
   Rebuild docker image without using cache.

--keep-alive
   Keep the container alive after the application finishes running.

--keep-model-dir
   Keep the model directory after the application finishes running.

--output, -o OUTPUT
   Output file for the result.

--log-level LOG_LEVEL
   Log level for the logger.

.. note::

   Learn more about MAD's capabilities by visiting the README at
   `<https://github.com/ROCm/MAD/blob/develop/README.md>`__.
