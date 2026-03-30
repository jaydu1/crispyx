crispyx
=======

Genome-wide CRISPR screens routinely produce datasets with hundreds of thousands
of cells and tens of thousands of genes. Standard single-cell toolkits load the
entire count matrix into memory, which can require 30–100+ GB of RAM. **crispyx**
streams data directly from on-disk AnnData ``.h5ad`` files so that quality
control, normalisation, pseudo-bulk aggregation, and differential expression all
run without materialising the full matrix — even the largest screens can be
processed with modest resources.

The API mirrors Scanpy (``cx.pp``, ``cx.pb``, ``cx.tl``, ``cx.pl``) so existing
workflows can migrate with minimal changes. See the :doc:`tutorial <crispyx_tutorial>`
for an end-to-end walkthrough.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   crispyx_tutorial
   benchmarking
   api
