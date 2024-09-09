<head>
  <meta charset="UTF-8">
  <meta name="description" content="Learn about how ROCm structures its documentation in its open-source projects.">
  <meta name="keywords" content="ROCm, contributing, contribute, maintainer, contributor, organization, restructuredtext">
</head>

# Documentation structure

Our documentation follows the Pitchfork folder structure. Most documentation files are stored in the
`/docs` folder. Some special files (such as release, contributing, and changelog) are stored in the root
(`/`) folder.

All images are stored in the `/docs/data` folder. An image's file path mirrors that of the documentation
file where it is used.

Our naming structure uses kebab case; for example, `my-file-name.rst`.

## Supported formats and syntax

Our documentation includes both Markdown and RST files. We are gradually transitioning existing
Markdown to RST in order to more effectively meet our documentation needs. When contributing,
RST is preferred; if you must use Markdown, use GitHub-flavored Markdown.

We use [Sphinx Design](https://sphinx-design.readthedocs.io/en/latest/index.html) syntax and compile
our API references using [Doxygen](https://www.doxygen.nl/).

The following table shows some common documentation components and the syntax convention we
use for each:

<table>
<tr>
<th>Component</th>
<th>RST syntax</th>
</tr>
<tr>
<td>Code blocks</td>
<td>

```rst

.. code-block:: language-name

  My code block.


```

</td>
</tr>
<tr>
<td>Cross-referencing internal files</td>
<td>

```rst

:doc:`Title <../path/to/file/filename>`

```

</td>
</tr>
<tr>
<td>External links</td>
<td>

```rst

`link name  <URL>`_

```

</td>
</tr>
<tr>
<tr>
<td>Headings</td>
<td>

```rst

******************
Chapter title (H1)
******************

Section title (H2)
===============

Subsection title (H3)
---------------------

Sub-subsection title (H4)
^^^^^^^^^^^^^^^^^^^^


```

</td>
</tr>
<tr>
<td>Images</td>
<td>

```rst

.. image:: image1.png

```

</td>
</tr>
<tr>
<td>Internal links</td>
<td>

```rst

1. Add a tag to the section you want to reference:

.. _my-section-tag: section-1

Section 1
==========

2. Link to your tag:

As shown in :ref:`section-1`.

```

</td>
</tr>
<tr>
<tr>
<td>Lists</td>
<td>

```rst

# Ordered (numbered) list item

* Unordered (bulleted) list item

```

</td>
</tr>
<tr>
<tr>
<td>Math (block)</td>
<td>

```rst

.. math::

  A = \begin{pmatrix}
          0.0 & 1.0 & 1.0 & 3.0 \\
          4.0 & 5.0 & 6.0 & 7.0 \\
        \end{pmatrix}

```

</td>
</tr>
<tr>
<td>Math (inline)</td>
<td>

```rst

:math:`2 \times 2 `

```

</td>
</tr>
<tr>
<td>Notes</td>
<td>

```rst

.. note::

  My note here.

```

</td>
</tr>
<tr>
<td>Tables</td>
<td>

```rst

.. csv-table::  Optional title here
  :widths: 30, 70  #optional column widths
  :header: "entry1 header", "entry2 header"

   "entry1", "entry2"

```

</td>
</tr>
</table>

## Language and style

We use the
[Google developer documentation style guide](https://developers.google.com/style/highlights) to
guide our content.

Font size and type, page layout, white space control, and other formatting
details are controlled via
[rocm-docs-core](https://github.com/ROCm/rocm-docs-core). If you want to notify us
of any formatting issues, create a pull request in our
[rocm-docs-core](https://github.com/ROCm/rocm-docs-core) GitHub repository.

## Building our documentation

<!--  % TODO: Fix the link to be able to work at every files  -->
To learn how to build our documentation, refer to
[Building documentation](./building.md).
