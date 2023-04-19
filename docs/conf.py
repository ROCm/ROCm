# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
shutil.copy2('../CONTRIBUTING.md','./contributing.md')

from rocm_docs import ROCmDocs

linkcheck_anchors_ignore = ['d90e61', 'd1667e113', 'd2999e60', 'building-from-source', 'use-the-rocm-build-tool-rbuild', 'use-cmake-to-build-migraphx', 'example']

docs_core = ROCmDocs("ROCm Docs 5.6.0 Alpha")
docs_core.setup()
docs_core.disable_main_doc_link()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
