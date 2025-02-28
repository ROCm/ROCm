# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil

shutil.copy2("../RELEASE.md", "./about/release-notes.md")

os.system("mkdir -p ../_readthedocs/html/downloads")
os.system("cp compatibility/compatibility-matrix-historical-6.0.csv ../_readthedocs/html/downloads/compatibility-matrix-historical-6.0.csv")

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "rocm.docs.amd.com")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

# configurations for PDF output by Read the Docs
project = "ROCm Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = "6.3.3"
release = "6.3.3"
setting_all_article_info = True
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""

# pages with specific settings
article_pages = [
    {"file": "about/release-notes", "os": ["linux"], "date": "2025-02-19"},
    {"file": "compatibility/compatibility-matrix", "os": ["linux"]},
    {"file": "compatibility/ml-compatibility/pytorch-compatibility", "os": ["linux"]},
    {"file": "compatibility/ml-compatibility/tensorflow-compatibility", "os": ["linux"]},
    {"file": "compatibility/ml-compatibility/jax-compatibility", "os": ["linux"]},
    {"file": "how-to/deep-learning-rocm", "os": ["linux"]},

    {"file": "how-to/rocm-for-ai/index", "os": ["linux"]},

    {"file": "how-to/rocm-for-ai/training/index", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/training/train-a-model", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/training/prerequisite-system-validation", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/training/train-a-model/benchmark-docker/megatron-lm", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/training/train-a-model/benchmark-docker/pytorch-training", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/training/scale-model-training", "os": ["linux"]},

    {"file": "how-to/rocm-for-ai/fine-tuning/index", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/fine-tuning/overview", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/fine-tuning/fine-tuning-and-inference", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/fine-tuning/single-gpu-fine-tuning-and-inference", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/fine-tuning/multi-gpu-fine-tuning-and-inference", "os": ["linux"]},

    {"file": "how-to/rocm-for-ai/inference/index", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference/install", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference/hugging-face-models", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference/llm-inference-frameworks", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference/vllm-benchmark", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference/deploy-your-model", "os": ["linux"]},

    {"file": "how-to/rocm-for-ai/inference-optimization/index", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/model-quantization", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/profiling-and-debugging", "os": ["linux"]},
    {"file": "how-to/rocm-for-ai/inference-optimization/workload", "os": ["linux"]},

    {"file": "how-to/system-optimization/index", "os": ["linux"]},
    {"file": "how-to/system-optimization/mi300x", "os": ["linux"]},
    {"file": "how-to/system-optimization/mi200", "os": ["linux"]},
    {"file": "how-to/system-optimization/mi100", "os": ["linux"]},
    {"file": "how-to/system-optimization/w6000-v620", "os": ["linux"]},
    {"file": "how-to/tuning-guides/mi300x/index", "os": ["linux"]},
    {"file": "how-to/tuning-guides/mi300x/system", "os": ["linux"]},
    {"file": "how-to/tuning-guides/mi300x/workload", "os": ["linux"]},
    {"file": "how-to/system-debugging", "os": ["linux"]},
    {"file": "how-to/gpu-enabled-mpi", "os": ["linux"]},
]

external_toc_path = "./sphinx/_toc.yml"

extensions = ["rocm_docs", "sphinx_reredirects", "sphinx_sitemap", "sphinxcontrib.datatemplates"]

external_projects_current_project = "rocm"

# Uncomment if facing rate limit exceed issue with local build 
# external_projects_remote_repository = ""

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "https://rocm-stg.amd.com/")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

html_static_path = ["sphinx/static/css", "data/how-to/rocm-for-ai/inference"]
html_css_files = ["rocm_custom.css", "rocm_rn.css", "vllm-benchmark.css"]
html_js_files = ["vllm-benchmark.js"]

html_title = "ROCm Documentation"

html_theme_options = {"link_main_doc": False}

redirects = {"reference/openmp/openmp": "../../about/compatibility/openmp.html"}

numfig = False
