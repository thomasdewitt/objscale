objscale documentation
======================

Object-based analysis functions for fractal dimensions and size distributions in atmospheric sciences and beyond. Optimized for large datasets.

Description
-----------

``objscale`` provides computational tools for analyzing the scaling properties of objects in 2D binary arrays. The package consolidates methods for calculating size distributions and fractal dimensions that account for finite domain effects and complex boundary conditions. Originally developed for atmospheric science applications, these methods apply broadly to any field where object scaling properties matter.

The package implements methods from two main papers:

- `DeWitt & Garrett (2024) <https://acp.copernicus.org/articles/24/8457/2024/>`_ - finite domain effects in size distributions  
- `DeWitt et al. (2025) <https://doi.org/10.5194/egusphere-2024-1498>`_ - fractal dimensions for cloud field characterization

Installation
------------

.. code-block:: bash

   pip install objscale

Quick Example
-------------

.. code-block:: python

   import objscale
   import numpy as np

   # Create binary array (e.g., cloud mask, percolation lattice)
   arrays = [(np.random.random((1000, 1000)) < 0.3).astype(int) for _ in range(4)]

   # Size distribution with finite domain corrections
   (exponent, error), (log10_sizes, log10_counts) = objscale.finite_array_powerlaw_exponent(
       arrays, 'area', return_counts=True
   )

   # Ensemble fractal dimensions
   corr_dim, corr_error = objscale.ensemble_correlation_dimension(arrays)
   box_dim, box_error = objscale.ensemble_box_dimension(arrays)

   # Individual object analysis  
   ind_dim, ind_error = objscale.individual_fractal_dimension(arrays)

Features
--------

- **Finite domain corrections**: Proper handling of truncation effects at domain boundaries
- **Multiple size metrics**: Area, perimeter, width, height, nested perimeter
- **Arbitrary boundaries**: Support for NaN-demarcated non-rectangular domains  
- **Individual and Ensemble methods**: Characterize both individual and collective properties of object fields
- **Performance optimized**: Numba acceleration for computational efficiency. Can handle billions of individual objects on a mid-range laptop.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/fractal_dimensions
   api/size_distributions
   api/object_analysis
   api/utilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`