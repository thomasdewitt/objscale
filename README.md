# objscale

Object-based analysis functions for fractal dimensions and size distributions in atmospheric sciences and beyond. Optimized for large datasets.

## Description

`objscale` provides computational tools for analyzing the scaling properties of objects in 2D binary arrays. The package consolidates methods for calculating size distributions and fractal dimensions that account for finite domain effects and complex boundary conditions. Originally developed for atmospheric science applications, these methods apply broadly to any field where object scaling properties matter.

The package implements methods from two main papers:

- [DeWitt & Garrett (2024)](https://acp.copernicus.org/articles/24/8457/2024/) - finite domain effects in size distributions  
- DeWitt et al. (in prep) - fractal dimensions for cloud field characterization

## Key Functions

### `finite_array_powerlaw_exponent`

Calculate power-law exponents for size distributions while accounting for finite domain truncation effects. Essential for accurate scaling analysis in bounded domains.

### `individual_fractal_dimension`

Fractal dimension of individual objects using the perimeter-area relationship, with proper handling of interior holes and resolution effects.

### `ensemble_correlation_dimension`

Correlation dimension for characterizing the collective scaling properties of object ensembles. Uses point-pair correlation analysis across multiple length scales.

### `ensemble_box_dimension`

Box-counting dimension for object ensembles. New analyses should prefer `ensemble_correlation_dimension`. Counts boxes containing object boundaries at varying spatial scales.

### `ensemble_coarsening_dimension`

Novel fractal dimension based on how total object perimeter changes under spatial coarsening operations. New analyses should prefer `ensemble_correlation_dimension`. Used in [Rees et. al., 2024](https://npg.copernicus.org/articles/31/497/2024/)

## Installation

```bash
pip install objscale
```

## Quick Example

```python
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
```

## Features

- **Finite domain corrections**: Proper handling of truncation effects at domain boundaries as recommended by [DeWitt & Garrett (2024)]([ACP - Finite domains cause bias in measured and modeled distributions of cloud sizes](https://acp.copernicus.org/articles/24/8457/2024/))
- **Multiple size metrics**: Area, perimeter, width, height, nested perimeter
- **Arbitrary boundaries**: Support for NaN-demarcated non-rectangular domains  
- **Individual and Ensemble methods**: Characterize both individual and collective properties of object fields
- **Performance optimized**: Numba acceleration for computational efficiency. Can handle billions of individual objects on a mid-range laptop.

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0  
- scikit-image ≥ 0.18.0
- Numba ≥ 0.56.0

## Support Statement

This package consolidates research code developed over several years. While functional and tested, it should be considered research software with limited ongoing support. Users are encouraged to understand the underlying methods through the referenced papers before applying to their data.

## References

If you use this package, please cite:

DeWitt, T. D. and Garrett, T. J.: Finite domains cause bias in measured and modeled 
distributions of cloud sizes, Atmos. Chem. Phys., 24, 8457–8472, 
https://doi.org/10.5194/acp-24-8457-2024, 2024.

## Author

Thomas D. DeWitt  
University of Utah Department of Atmospheric Sciences

## License

MIT License