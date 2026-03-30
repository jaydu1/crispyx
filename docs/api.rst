API Reference
=============

crispyx exposes a Scanpy-style API through four namespace singletons:

* ``cx.pp`` — Preprocessing (QC, normalisation, PCA, neighbours, format conversion)
* ``cx.pb`` — Pseudo-bulk aggregation
* ``cx.tl`` — Tools (differential expression, LFC shrinkage, overlap analysis)
* ``cx.pl`` — Plotting

Most functions also accept file paths or backed AnnData objects directly.

Namespace API
-------------

Preprocessing (``cx.pp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PreprocessingNamespace
   :members:
   :undoc-members:

Pseudo-bulk (``cx.pb``)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PseudobulkNamespace
   :members:
   :undoc-members:

Tools (``cx.tl``)
~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._ToolsNamespace
   :members:
   :undoc-members:

Plotting (``cx.pl``)
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: crispyx._namespaces._PlottingNamespace
   :members:
   :undoc-members:


Module Reference
----------------

The sections below document the underlying modules used by the namespace
API.  These are useful for advanced usage or for understanding parameter
details.

Data loading and utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.data
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Quality control
~~~~~~~~~~~~~~~

.. automodule:: crispyx.qc
   :members:
   :undoc-members:
   :show-inheritance:

Pseudo-bulk aggregation
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.pseudobulk
   :members:
   :undoc-members:
   :show-inheritance:

Differential expression
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.de
   :members:
   :undoc-members:
   :show-inheritance:

Negative binomial GLM
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.glm
   :members:
   :undoc-members:
   :show-inheritance:

Dimension reduction
~~~~~~~~~~~~~~~~~~~

.. automodule:: crispyx.dimred
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

.. automodule:: crispyx.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Profiling
~~~~~~~~~

.. automodule:: crispyx.profiling
   :members:
   :undoc-members:
   :show-inheritance:
