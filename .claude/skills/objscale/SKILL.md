---
name: objscale
description: Use when working with objscale package for analyzing 2D binary arrays - fractal dimensions, size distributions, object scaling properties. Also use for cloud masks, percolation lattices, segmentation masks, or any task involving connected components in 2D binary data.
---

# objscale Package Reference

Object-based analysis functions for fractal dimensions and size distributions in **2D binary arrays**.

**Documentation**: https://objscale.readthedocs.io

## Critical Usage Note

**When analyzing multiple arrays (e.g., multiple cloud field images), pass them ALL AT ONCE as a list.** This applies to all functions that accept lists of arrays: fractal dimensions (ensemble and individual), size distributions, etc. Do not call functions separately for each array and combine results - these are not linear operations.

```python
# CORRECT - pass all arrays at once
arrays = [array1, array2, array3, array4]
dim, err = objscale.ensemble_correlation_dimension(arrays)
ind_dim, err = objscale.individual_fractal_dimension(arrays)
exp, err = objscale.finite_array_powerlaw_exponent(arrays, 'area')

# WRONG - do not loop and combine
# for arr in arrays:
#     result = some_objscale_function(arr)  # NO!
```

## When to Use Which Function

### Fractal Dimensions

| Task | Function | Notes |
|------|----------|-------|
| **Recommended** ensemble fractal dimension | `ensemble_correlation_dimension` | Most robust method |
| Individual object fractal dimension | `individual_fractal_dimension` | Uses perimeter-area scaling |
| Box-counting dimension | `ensemble_box_dimension` | Classic method, prefer correlation |

**Note**: Do not use `ensemble_coarsening_dimension` - it has ambiguity issues with binary array coarsening.

### Size Distributions

| Task | Function | Notes |
|------|----------|-------|
| Power-law exponent | `finite_array_powerlaw_exponent` | **Always prefer this** - accounts for domain truncation |
| Full distribution with truncation info | `finite_array_size_distribution` | Returns truncated/non-truncated counts |
| Simple distribution (no corrections) | `array_size_distribution` | Only for special cases |

**IMPORTANT**: Always use `finite_*` functions for size distributions. They properly account for objects truncated by domain boundaries.

**For plotting**: Use `finite_array_powerlaw_exponent` with `return_counts=True` to get log10 bin middles and counts suitable for plotting the distribution.

### Object Analysis

| Task | Function |
|------|----------|
| Get perimeter, area, width, height | `get_structure_props` |
| Total perimeter of all objects | `total_perimeter` |
| Count objects | `total_number` |
| Extract largest object | `isolate_largest_structure` |
| Remove border-touching objects | `remove_structures_touching_border_nan` |
| Fill holes in objects | `remove_structure_holes` |
| Label objects by size | `label_size` |
| Clear border-adjacent structures | `clear_border_adjacent` |

### Utilities

| Task | Function |
|------|----------|
| Reduce array resolution | `coarsen_array` |
| Linear regression with errors | `linear_regression` |
| Add border to array | `encase_in_value` |

## Function Signatures

### Fractal Dimensions

```python
objscale.ensemble_correlation_dimension(
    arrays,                      # Binary arrays (list or single)
    x_sizes=None,                # Pixel sizes in x (2D array, same shape as arrays)
    y_sizes=None,                # Pixel sizes in y
    minlength='auto',            # Min scale (default: 3x pixel size)
    maxlength='auto',            # Max scale (default: 0.33x domain size)
    interior_circles_only=True,  # Avoid boundary effects (recommended!)
    return_C_l=False,            # Return (dim, err, bins, C_l)
    bins=None,                   # Custom bin edges, or int for number of bins
    point_reduction_factor=1,    # Subsample points (>=1, for speed)
    nbins=50                     # Number of scale bins when bins=None
) -> (dimension, error) | (dimension, error, bins, C_l)
```

```python
objscale.individual_fractal_dimension(
    arrays,                # Binary arrays (list or single)
    x_sizes=None,          # Pixel sizes in x
    y_sizes=None,          # Pixel sizes in y
    min_a=10,              # Min area to include
    max_a=np.inf,          # Max area to include
    return_values=False    # Return (dim, err, log10_sqrt_a, log10_p)
) -> (dimension, error) | (dimension, error, log10_sqrt_a, log10_p)
```

```python
objscale.ensemble_box_dimension(
    binary_arrays,         # Binary arrays (list or single)
    set='edge',            # 'edge' (boundaries) or 'ones' (all 1s)
    min_pixels=1,          # Largest box size constraint
    min_box_size=2,        # Smallest box size
    box_sizes='default',   # Custom box sizes or 'default' (powers of 2)
    return_values=False    # Return (dim, err, box_sizes, counts)
) -> (dimension, error) | (dimension, error, box_sizes, mean_counts)
```

### Size Distributions

```python
objscale.finite_array_powerlaw_exponent(
    arrays,                    # Binary arrays (list or single)
    variable,                  # 'area', 'perimeter', 'width', 'height', 'nested perimeter'
    x_sizes=None,              # Pixel sizes in x
    y_sizes=None,              # Pixel sizes in y
    bins=100,                  # Number of bins or array of log10(bin edges)
    min_threshold=10,          # Minimum object size
    truncation_threshold=0.5,  # Max fraction of truncated objects per bin
    min_count_threshold=30,    # Min objects per bin for regression
    return_counts=False        # Return ((exp, err), (log10_bins, log10_counts))
) -> (exponent, error) | ((exponent, error), (log10_bins, log10_counts))
```

```python
objscale.finite_array_size_distribution(
    arrays,                    # Binary arrays (list or single)
    variable,                  # 'area', 'perimeter', 'width', 'height', 'nested perimeter'
    x_sizes=None,              # Pixel sizes in x
    y_sizes=None,              # Pixel sizes in y
    bins=100,                  # Number of bins or array of bin edges
    bin_logs=True,             # If True, bins are log-spaced
    min_threshold=10,          # Minimum bin edge
    truncation_threshold=0.5   # Threshold for truncation dominance
) -> (bin_middles, nontruncated_counts, truncated_counts, truncation_index)
```

```python
objscale.array_size_distribution(
    array,                 # Single binary array
    variable='area',       # 'area', 'perimeter', 'width', 'height', 'nested perimeter'
    bins=30,               # Number of bins or array of bin edges
    bin_logs=True,         # If True, bin log10(variable)
    structure=...,         # Connectivity structure (default: 4-connected)
    wrap=None,             # 'sides', 'both', or None
    x_sizes=None,          # Pixel sizes in x
    y_sizes=None           # Pixel sizes in y
) -> (bin_middles, counts)
```

### Object Analysis

```python
objscale.get_structure_props(
    array,                 # Binary array (0s, 1s, nans)
    x_sizes,               # Pixel sizes in x (same shape as array)
    y_sizes,               # Pixel sizes in y
    structure=...,         # Connectivity (default: 4-connected)
    print_none=False,      # Print message if no structures
    wrap=None              # 'sides', 'both', or None for periodic boundaries
) -> (perimeters, areas, heights, widths)  # Each is 1D array, one per structure
```

```python
objscale.total_perimeter(
    array,     # Binary array (0s and 1s)
    x_sizes,   # Pixel sizes in x
    y_sizes    # Pixel sizes in y
) -> float   # Total perimeter length
```

```python
objscale.total_number(
    array,           # Binary array (0s, 1s, nans)
    structure=...    # Connectivity (default: 4-connected)
) -> int   # Number of connected objects
```

```python
objscale.isolate_largest_structure(
    binary_array,    # Binary input array
    structure=...    # Connectivity (default: 4-connected)
) -> np.ndarray[bool]   # Boolean array with only largest structure True
```

```python
objscale.label_size(
    array,             # Binary array
    variable='area',   # 'area', 'perimeter', 'width', 'height'
    wrap='both',       # 'sides', 'both', or None
    x_sizes=None,      # Pixel sizes in x
    y_sizes=None       # Pixel sizes in y
) -> np.ndarray   # Array where each structure labeled with its size value
```

```python
objscale.remove_structures_touching_border_nan(
    array    # 2D array of 0s, 1s, nans (edges should be nan)
) -> np.ndarray   # Array with border-touching structures removed
```

```python
objscale.remove_structure_holes(
    array,           # 2D array of 0s, 1s, nans
    periodic=False   # False, 'sides', or 'both'
) -> np.ndarray   # Array with all holes in structures filled
```

```python
objscale.clear_border_adjacent(
    array,          # 2D array of 0s and 1s
    structure=...   # Connectivity (default: 4-connected)
) -> np.ndarray[bool]   # Array with border-touching structures removed
```

### Utilities

```python
objscale.coarsen_array(
    array,    # Input array
    factor    # Coarsening factor (int)
) -> np.ndarray   # Array reduced by factor via averaging
```

```python
objscale.linear_regression(
    x,   # Independent variable (1D array)
    y    # Dependent variable (1D array)
) -> ((slope, intercept), (slope_error, intercept_error))  # 95% CI errors
```

```python
objscale.encase_in_value(
    array,            # 2D input array
    value=np.nan,     # Border value
    dtype=np.float32, # Output dtype
    n_deep=1          # Border thickness
) -> np.ndarray   # Array with border added
```

## Handling Non-Rectangular Domains

Use `np.nan` to mark regions outside your domain of interest:
- Objects touching nan boundaries are treated as truncated
- Perimeter is not counted along nan boundaries
- Enables arbitrary domain shapes

```python
array = np.random.random((500, 500)) < 0.3
array = array.astype(float)
array[:100, :] = np.nan  # Mark region as outside domain
```

## Variable-Size Pixels

For non-uniform grids (e.g., lat/lon data), pass pixel size arrays:

```python
x_sizes = np.ones((ny, nx)) * dx  # Can vary spatially
y_sizes = np.ones((ny, nx)) * dy

dim, err = objscale.ensemble_correlation_dimension(
    arrays, x_sizes=x_sizes, y_sizes=y_sizes
)
```

## References

If using this package, cite:

### DeWitt & Garrett (2024) - Size Distributions

**Finite domains cause bias in measured and modeled distributions of cloud sizes.**
*Atmos. Chem. Phys.* https://doi.org/10.5194/acp-24-8457-2024

> A significant uncertainty in assessments of the role of clouds in climate is the characterization of the full distribution of their sizes. Order-of-magnitude disagreements exist among observations of key distribution parameters, particularly power law exponents and the range over which they apply. A study by Savre and Craig (2023) suggested that the discrepancies are due in large part to inaccurate fitting methods: they recommended the use of a maximum likelihood estimation technique rather than a linear regression to a logarithmically transformed histogram of cloud sizes. Here, we counter that linear regression is both simpler and equally accurate, provided the simple precaution is followed that bins containing fewer than ~24 counts are omitted from the regression. A much more significant and underappreciated source of error is how to treat clouds that are truncated by the edges of unavoidably finite measurement domains. We offer a simple computational procedure to identify and correct for domain size effects, with potential application to any geometric size distribution of objects, whether physical, ecological, social or mathematical.

### DeWitt et al. (2025) - Fractal Dimensions

**Fractal dimensions for cloud field characterization.**
*EGUsphere* https://doi.org/10.5194/egusphere-2025-3486

> As clouds sizes and shapes become better resolved by numerical climate models, objective metrics are required to evaluate whether simulations satisfactorily reflect observations. However, even the most recent cloud classification schemes rely on quite subjectively defined visual categories that lack any direct connection to the underlying physics. The fractal dimension of cloud fields has been used to provide a more objective footing. But, as we describe here, there are a wide range of largely unrecognized subtleties to such analyses that must be considered prior to obtaining meaningfully quantitative results. Methods are described for calculating two distinct types of fractal dimension: an individual fractal dimension Di representing the roughness of individual cloud edges, and an ensemble fractal dimension De characterizing how cloud fields organize hierarchically across spatial scales. Both have the advantage that they can be linked to physical symmetry principles, but De is argued to be better suited for observational validation of simulated collections of clouds, particularly when it is calculated using a straightforward correlation integral method. A remaining challenge is an observed sensitivity of calculated values of De to subjective choices of the reflectivity threshold used to distinguish clouds from clear skies. We advocate that, in the interests of maximizing objectivity, future work should consider treating cloud ensembles as continuous reflectivity fields rather than collections of discrete objects.
