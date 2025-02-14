<head>
  <meta charset="UTF-8">
  <meta name="description" content="ROCm documentation toolchain">
  <meta name="keywords" content="documentation, toolchain, Sphinx, Doxygen, MyST, AMD, ROCm">
</head>

# ROCm documentation toolchain

The ROCm documentation relies on several open source toolchains and sites.

## rocm-docs-core

[rocm-docs-core](https://github.com/ROCm/rocm-docs-core) is an AMD-maintained
project that applies customizations for the ROCm documentation. This project is the tool most ROCm repositories use as part of their documentation build pipeline. It is available as a [pip package on PyPI](https://pypi.org/project/rocm-docs-core/).

See the user and developer guides for rocm-docs-core at
{doc}`rocm-docs-core documentation<rocm-docs-core:index>`.

## Sphinx

[Sphinx](https://www.sphinx-doc.org/en/master/) is a documentation generator originally used for Python. It is now widely used in the open source community.

### Sphinx External ToC

[Sphinx External ToC](https://sphinx-external-toc.readthedocs.io/en/latest/intro.html) is a Sphinx extension used for ROCm documentation navigation. This tool generates a navigation menu on the left
based on a YAML file (`_toc.yml.in`) that contains the table of contents.

### Sphinx-book-theme

[Sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/en/latest/) is a Sphinx theme that defines the base appearance for ROCm documentation. ROCm documentation applies some customization, such as a custom header and footer, on top of the Sphinx Book Theme.

### Sphinx Design

[Sphinx design](https://sphinx-design.readthedocs.io/en/latest/index.html) is a Sphinx extension that adds design functionality. ROCm documentation uses Sphinx Design for grids, cards, and synchronized tabs.

## Doxygen

[Doxygen](https://www.doxygen.nl/) is a documentation generator that extracts information from in-code comments. It is used for API documentation.

## Breathe

[Breathe](https://www.breathe-doc.org/) is a Sphinx plugin for integrating Doxygen content.

## Read the Docs

[Read the Docs](https://docs.readthedocs.io/en/stable/) is the service that builds and hosts the HTML version of the ROCm documentation.
