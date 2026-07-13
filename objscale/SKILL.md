---
name: objscale
description: Use when working with objscale package for analyzing 2D binary arrays - fractal dimensions, size distributions, object scaling properties. Also use for cloud masks, percolation lattices, segmentation masks, or any task involving connected components in 2D binary data.
---

# objscale Package Reference

Object-based analysis functions for fractal dimensions and size distributions in **2D binary arrays**. Version **2.0.0**.

**Version check**: If you happen to determine that the installed objscale version does not match the version above, this skill may be outdated. For objscale >= 2.0.0, the skill matching the installed version is bundled with the package itself: read it at the path returned by `objscale.skill_path()`, or install it with `objscale.install_agent_skill('claude')` (or `'codex'`). You can also fetch the latest development copy from https://raw.githubusercontent.com/thomasdewitt/objscale/master/objscale/SKILL.md and save it to your local skill folder.

**Documentation**: https://objscale.readthedocs.io

**Background reference**: Many of the recommendations in this skill rest on subtle issues in estimating scaling exponents from finite, discretized data — finite-domain bias, the multiplicity of distinct "fractal dimensions", fitting-range choices, and uncertainty calibration. These are treated carefully, with empirical demonstrations, in the blog post ["Too many exponents"](https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html). If you hit results that look wrong, non-power-law scaling, or disagreements between estimators, fetch that post — it likely discusses your situation.

## Critical Usage Notes

### Pass all arrays at once

**When analyzing multiple arrays (e.g., multiple cloud field images), pass them ALL AT ONCE as a list.** This applies to all functions that accept lists of arrays: fractal dimensions (ensemble and individual), size distributions, etc. Do not call functions separately for each array and combine results - these are not, in general, linear operations.

```python
# CORRECT - pass all arrays at once
arrays = [array1, array2, array3, array4]
dim = objscale.ensemble_correlation_dimension(arrays)
ind_dim = objscale.individual_fractal_dimension(arrays)
exp = objscale.finite_array_powerlaw_exponent(arrays, 'area')

# WRONG - do not loop and combine
# for arr in arrays:
#     result = some_objscale_function(arr)  # NO!
```

If you are attempting to analyze a large dataset that does not fit in memory, you MUST carefully consider whether the specific function you are using is linear. In the above example, the output dimensions/exponents are computed using a linear regression and are NOT linear with the inputs. Other functions, such as bin counts from a size distribution, can sometimes return linear outputs. You may have to compute the regression yourself. You MUST carefully consider cases where loops are necessary (is the function linear? is each input chunk the same size? does `nan` prevalence change?), and when you are unsure explain the situation to your user!

### No uncertainty estimates are returned — deliberately

As of v2.0.0, all exponent/dimension estimators return point estimates only (in versions < 2.0.0 they returned `(value, uncertainty)` tuples). The removed uncertainties were 2× the OLS standard error of the log-log regression, and they were **miscalibrated to the point of being misleading**: OLS standard errors assume the residuals at each scale are statistically independent, but the points of a scaling function derived from a fractal or multifractal field are strongly correlated across scales (a single large object influences many bins at once). Bootstrapping against truly independent realizations shows the OLS-derived uncertainty can be far too small for small sample sizes and even too large for big ensembles — see the ["Statistical error and parameter uncertainty" section of "Too many exponents"](https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html) for the demonstration. Do not report an uncertainty for these exponents unless you compute a defensible one yourself.

A viable alternative exists, but you must do it yourself, deliberately: **bootstrap across statistically independent images** — resample whole arrays (with replacement) from your ensemble, recompute the exponent per resample, and take the spread. Critically, resampling must be across *images*, not across objects or bins within one image (those are not independent). This requires that your images really are statistically independent realizations, which is your responsibility to establish. If you have only a single image, there is no honest uncertainty estimate available; say so rather than inventing one.

`objscale.linear_regression` still returns regression standard errors: it is a generic utility, and its errors are valid for genuinely independent data — just not for scaling functions of scale-invariant fields.

## When to Use Which Function

### Fractal Dimensions

| Task                                             | Function                           | Notes                                                                      |
| ------------------------------------------------ | ---------------------------------- | -------------------------------------------------------------------------- |
| **Recommended** ensemble fractal dimension (q=2) | `ensemble_correlation_dimension`   | Thin wrapper around `ensemble_sandbox_renyi_dimension(q=2.0)`              |
| Generalized Rényi dimension D_q via sandbox      | `ensemble_sandbox_renyi_dimension` | Set-centered balls at continuous radii. Best for any q except q=0.         |
| Generalized Rényi dimension D_q via box counting | `ensemble_box_renyi_dimension`     | Fixed-grid tiles. Natural for q=0; noisier than sandbox at q≠0.            |
| Box-counting dimension (q=0)                     | `ensemble_box_dimension`           | Box-Renyi at q=0.                                                          |
| Information dimension (q=1)                      | `ensemble_information_dimension`   | Defaults to `method='sandbox'`; pass `method='box'` for the box estimator. |
| Correlation dimension of a single object         | `individual_correlation_dimension` | Isolates Nth largest structure, computes correlation dim                   |
| Individual object fractal dimension              | `individual_fractal_dimension`     | Perimeter vs. area/width/height scaling, selectable via `method=` strings  |

**Two estimators, one quantity.** Box and sandbox both estimate the same Rényi dimension D_q, but sample the partition function differently. Box counting tiles space with a fixed grid; sandbox places balls on set points and counts neighbors at continuous radii. They agree for monofractal sets and disagree only at finite-size scales. Use sandbox by default; box is the natural choice only for q=0 (where sandbox uses an awkward inverse-mass form). In general, sandbox and higher q are less sensitive to discretization bias as discussed in Too Many Exponents.

### Size Distributions

| Task                                   | Function                         | Notes                                                   |
| -------------------------------------- | -------------------------------- | ------------------------------------------------------- |
| Power-law exponent                     | `finite_array_powerlaw_exponent` | **Always prefer this** - accounts for domain truncation |
| Full distribution with truncation info | `finite_array_size_distribution` | Returns truncated/non-truncated counts                  |
| Simple distribution (no corrections)   | `array_size_distribution`        | Only for special cases                                  |

**IMPORTANT**: Always use `finite_*` functions for size distributions. They properly account for objects truncated by domain boundaries.

**For plotting**: Use `finite_array_powerlaw_exponent` with `return_counts=True` to get log10 bin middles and counts suitable for plotting the distribution.

**Theory — finite domain truncation (DeWitt & Garrett 2024)**: Objects in finite domains can be *truncated* (touching the domain boundary) or *non-truncated* (entirely interior). Truncated objects have unknown true sizes because the portion beyond the boundary is unmeasured. Larger objects are more likely to be truncated, so the large end of a size distribution is progressively contaminated. The `finite_*` functions correct for this by:

1. Binning sizes in log-space and separately counting truncated vs. non-truncated objects per bin.
2. Excluding bins where the fraction of truncated objects exceeds `truncation_threshold` (default 0.5). This removes the large-size tail where most objects are cut off by the domain edge.
3. Excluding bins with fewer than `min_count_threshold` counts (default 30). Below ~24 counts, bin-count statistics are non-Gaussian and least-squares regression on log-transformed counts becomes unreliable.
4. Fitting a power law via linear regression to the surviving "good" bins.

`finite_array_size_distribution` returns all four arrays — `(bin_middles, nontruncated_counts, truncated_counts, truncation_index)` — so you can inspect which bins survived and plot truncated vs. non-truncated counts separately. `truncation_index` is the first bin index where the truncation fraction exceeds the threshold. `finite_array_powerlaw_exponent` does steps 1–4 internally and returns the fitted exponent directly.

### Object Analysis

| Task                                        | Function                                |
| ------------------------------------------- | --------------------------------------- |
| Label connected components                  | `label_structures`                      |
| Get structure areas (fast, O(n))            | `get_structure_areas`                   |
| Get structure perimeters (fast, O(n))       | `get_structure_perimeters`              |
| Get structure height and width              | `get_structure_height_width`            |
| Get all four properties at once             | `get_structure_props`                   |
| Get every boundary perimeter (incl. nested) | `get_every_boundary_perimeter`          |
| Total perimeter of all objects              | `total_perimeter`                       |
| Count objects                               | `total_number`                          |
| Extract Nth largest object                  | `isolate_nth_largest_structure`         |
| Remove border-touching objects              | `remove_structures_touching_border_nan` |
| Fill holes in objects                       | `remove_structure_holes`                |
| Label objects by size                       | `label_size`                            |
| Clear border-adjacent structures            | `clear_border_adjacent`                 |

### Utilities

| Task                             | Function                         |
| -------------------------------- | -------------------------------- |
| Reduce array resolution          | `coarsen_array`                  |
| Linear regression with errors    | `linear_regression`              |
| Add border to array              | `encase_in_value`                |
| Find boundary pixel coordinates  | `get_coords_of_boundaries`       |
| Convert pixel sizes to locations | `get_locations_from_pixel_sizes` |
| Set number of parallel threads   | `set_num_threads`                |
| Install this skill for an agent  | `install_agent_skill`            |
| Path to the bundled skill file   | `skill_path`                     |

## Interpreting Results: Scaling Checks and Finite-Size Bias

These estimators return a single number, but that number is only meaningful if the underlying statistic actually follows a power law over the fitted range. Smart usage requires checking this, and understanding *why* it can fail. When you report results from these functions, mention these caveats to your user where they are relevant.

**The non-power-law sanity check.** A fractal dimension or size-distribution exponent is *defined* through a power law. If the local exponent (the local slope of the scaling function in log-log space) drifts systematically with scale, the data is not scale invariant over that range, and the fitted number should not be reported as a fractal dimension or power-law exponent at all — for example, reporting that "the fractal dimension increases with object size" is a category error; a scale-dependent "dimension" is not a dimension. Always inspect the scaling function when results matter: pass `return_C_l=True` / `return_values=True` / `return_counts=True`, plot log10(statistic) vs. log10(scale), and check that the fitted range is actually linear.

**Deviations near the pixel and domain scales are usually discretization bias, not physics.** Both the pixels and the overall domain are Euclidean shapes (rectangles) with well-defined scales; they break scale invariance in the *measurement* even when the underlying field is perfectly scale invariant. In practice, nontrivial bias can persist to scales orders of magnitude away from the pixel or domain scale — even theoretically scale-invariant simulations at 8192² still show clear curvature near both ends of the scaling range, and this depends on the specific scaling function considered (i.e. correlation dimension is scaling to smaller scales relative to  box dimension). So interpret carefully: local-slope drift confined to the smallest and largest scales means you should restrict the fitting range (see below), while drift throughout the range means the data may genuinely not be scale invariant. Both of these can be true at once: the phenomenon is scale invariant *and* the measurement is not. Do not conclude "not a fractal" from curvature near the resolution or domain limits alone.

**Fitting-range choice is a real (and unavoidably subjective) degree of freedom.** The default fitting ranges (e.g., 8× pixel scale to 0.33× domain scale for sandbox dimensions) are heuristics that balance bias against fitting a wide range of scales. When accuracy against a known or comparable value matters, empirical tests on simulations with theoretically-known exponents show that restricting the regression to the log-central ~25% of the available scaling range measurably reduces bias relative to fitting the full range — at the cost of no longer verifying power-law behavior over a wide range. Importantly, the 25% heuristic requires knowing that the underlying data are scaling in the first place. Report the fitting range used; two studies fitting different ranges will get different exponents from identical data.

**Distinct dimensions are distinct.** Box vs. sandbox estimators, ensemble vs. individual dimensions, filled vs. unfilled/summed perimeters, different Rényi orders q, different thresholds used to binarize a continuous field — these define *different exponents with different values*, even for the same underlying data, and many converge to each other only in the infinite-domain limit. Never compare a value computed one way against a literature value computed another way as if they estimate the same quantity. When reporting, state precisely which definition was used (function, method/set options, q, threshold, fitting range).

For the empirical demonstrations behind all of the above — convergence comparisons between estimators, bias persisting far from the pixel scale, fitting-window tests, and the zoo of inequivalent dimensions — fetch ["Too many exponents"](https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html).

## Function Signatures

### Fractal Dimensions

```python
objscale.ensemble_correlation_dimension(
    arrays,                      # Binary arrays (list or single)
    x_sizes=None,                # Pixel sizes in x (2D array, same shape as arrays)
    y_sizes=None,                # Pixel sizes in y
    minlength='auto',            # Min scale (default: 8x pixel size)
    maxlength='auto',            # Max scale (default: 0.33x domain size)
    interior_circles_only=False, # If True, only use centers >=maxlength from edges
    return_C_l=False,            # Return (dim, bins, C_l)
    bins=None,                   # Custom bin edges, or int for number of bins
    point_reduction_factor=1,    # Subsample points (>=1, for speed)
    nbins=50                     # Number of scale bins when bins=None
) -> dimension | (dimension, bins, C_l)
```

Thin wrapper around `ensemble_sandbox_renyi_dimension(..., q=2.0, set='edge')`. The sandbox q=2 partition function `sum_i M_i(r)` is exactly the Grassberger-Procaccia correlation integral, so this name is preserved as the standard q=2 entry point.

`interior_circles_only` defaults to `False`; see ["Correlation dimension and domain boundary effects"](https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html) for the rationale.

```python
objscale.ensemble_sandbox_renyi_dimension(
    binary_arrays,               # Binary arrays (list or single)
    q=2.0,                       # Rényi order(s); scalar or 1-D array
    set='edge',                  # 'edge' (one-sided edge mask) or 'ones'
    x_sizes=None,                # Pixel sizes in x
    y_sizes=None,                # Pixel sizes in y
    minlength='auto',            # Min scale
    maxlength='auto',            # Max scale
    interior_circles_only=False, # If True, only use centers >=maxlength from edges
    nbins=50,                    # Number of scale bins when bins=None
    bins=None,                   # Custom bin edges or int
    point_reduction_factor=1,    # Subsample sandbox centers (>=1)
    return_values=False          # Return (D_q, bins, Z)
) -> D_q | (D_q, bins, Z)
```

Sandbox-method Rényi dimension D_q. For each set point, count neighbors `M_i(r)` within radius r; partition function is `Z_q(r) = sum_i M_i^(q-1)` for `q != 1` (slope vs log r divided by `q-1` gives `D_q`), or per-center mean `<log10 M(r)>` for `q == 1` (slope is `D_1` directly). Strict generalization of Grassberger-Procaccia (`q=2` is the pair count). Better than box counting at `q != 0`; for `q = 0` use `ensemble_box_dimension` because sandbox q=0 needs the inverse-mass form. Reference: Tél, Fülöp, Vicsek 1989, Physica A 159; Vicsek 1992 textbook ch. 3.

```python
objscale.individual_correlation_dimension(
    array,                       # Single 2D binary array
    n=1,                         # Which structure (1=largest after border removal)
    x_sizes=None,                # Pixel sizes in x
    y_sizes=None,                # Pixel sizes in y
    minlength='auto',            # Min scale
    maxlength='auto',            # Max scale
    return_C_l=False,            # Return (dim, bins, C_l)
    point_reduction_factor=1,    # Subsample points (>=1)
    nbins=50,                    # Number of scale bins
    filled=True                  # Fill interior holes before computing (recommended)
) -> dimension | (dimension, bins, C_l)
```

```python
objscale.individual_fractal_dimension(
    arrays,                # Binary arrays (list or single)
    x_sizes=None,          # Pixel sizes in x
    y_sizes=None,          # Pixel sizes in y
    min_length_scale=3,    # Min length scale (sqrt-area/width/height) to include
    max_length_scale=np.inf,  # Max length scale to include
    bins=30,               # Number of bins for averaging (None = no binning)
    return_values=False,   # Return (dim, log10_length_scale, log10_p)
    method='filled perimeter vs filled area'  # See options below
) -> dimension | (dimension, log10_length_scale, log10_p)
```

`method` selects which perimeter and length-scale combination defines the dimension — these are *distinct exponents with distinct values* (see [Interpreting Results](#interpreting-results-scaling-checks-and-finite-size-bias)). Options: `'filled perimeter vs filled area'` (default, recommended — holes must be filled for the perimeter-area relation to represent a true boundary fractal dimension, see DeWitt et al. 2026), `'summed perimeter vs unfilled area'`, `'filled perimeter vs width'`, `'filled perimeter vs height'`, `'summed perimeter vs width'`, `'summed perimeter vs height'`. (Deprecated params `filled`, `min_a`, `max_a` map onto these.)

```python
objscale.ensemble_box_renyi_dimension(
    binary_arrays,         # Binary arrays (list or single)
    q=0.0,                 # Rényi order(s); scalar or 1-D array
    set='edge',            # 'edge' (one-sided edge mask: 1-pixels with a 0-neighbor) or 'ones'
    box_sizes='default',   # Custom box sizes or 'default' (powers of 2)
    max_box_size=None,     # Largest box in pixels (None = min(arr.shape))
    min_box_size=8,        # Smallest box in pixels
    box_origin_shift=(0.0, 0.0),  # Fractional (sx, sy) shift of box grid origin
    return_values=False    # Return (D_q, box_sizes, partition)
) -> D_q | (D_q, box_sizes, partition)
```

Box-counting Rényi dimension D_q. Always uses interior-only boxes (input is trimmed to a multiple of the current box size). For q != 1 the normalization is geometric (n_i / V where V is total interior pixel area); for q == 1 the Shannon entropy form is used. Use this for q=0 (where it's the natural box-counting dimension) and as a robustness comparison against the sandbox method.

```python
objscale.ensemble_box_dimension(
    binary_arrays,         # Binary arrays (list or single)
    set='edge',            # 'edge' or 'ones'
    max_box_size=None,     # Largest box in pixels (None = min(arr.shape))
    min_box_size=8,        # Smallest box in pixels
    box_sizes='default',   # Custom box sizes or 'default' (powers of 2)
    return_values=False    # Return (D_0, box_sizes, n_boxes)
) -> D_0 | (D_0, box_sizes, n_boxes)
```

Thin wrapper around `ensemble_box_renyi_dimension(..., q=0)`. Box counting is the natural q=0 estimator (sandbox at q=0 uses an awkward inverse-mass form).

```python
objscale.ensemble_information_dimension(
    binary_arrays,         # Binary arrays (list or single)
    method='sandbox',      # 'sandbox' (default) or 'box'
    set='edge',            # 'edge' or 'ones'
    return_values=False,   # Return (D_1, bins, partition)
    **kwargs,              # Method-specific options forwarded to underlying function
) -> D_1 | (D_1, bins, partition)
```

Information dimension D_1 (the q=1 Rényi dimension). Dispatches to `ensemble_sandbox_renyi_dimension(q=1)` by default, or `ensemble_box_renyi_dimension(q=1)` with `method='box'`. Sandbox is the default because it has lower grid-quantization noise at q=1. Method-specific options (`minlength`, `maxlength`, ... for sandbox; `max_box_size`, `box_sizes`, ... for box) are forwarded via `**kwargs`.

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
    return_counts=False        # Return (exponent, (log10_bins, log10_counts))
) -> exponent | (exponent, (log10_bins, log10_counts))
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
objscale.get_every_boundary_perimeter(
    array,                 # 2D binary array (0s and 1s)
    x_sizes,               # Pixel sizes in x
    y_sizes,               # Pixel sizes in y
    return_nlevels=False   # Also return number of nesting levels
) -> list | (list, int)   # Perimeters of each boundary (incl. nested holes)
```

```python
objscale.label_structures(
    array,                 # 2D binary array (0s, 1s, nans)
    structure=...,         # Connectivity (default: 4-connected)
    wrap='both'            # 'both', 'sides', or None (periodic boundary merging)
) -> (labelled_array, nan_mask, n_labels)  # or (None, None, 0) if empty
# labelled_array: float32, each positive value is a label. NaN pixels are 0.
# nan_mask: bool array of where NaN was in the input.
# n_labels: int count of labels.
```

```python
# NOTE: get_structure_areas/perimeters/height_width are lower-level functions
# that require a pre-labelled array from label_structures(). They return
# arrays of shape (n_labels,) indexed by label, guaranteeing alignment
# across metrics (area[i] corresponds to perimeter[i]).

objscale.get_structure_areas(
    labelled_array,        # From label_structures()
    nan_mask,              # From label_structures()
    n_labels,              # From label_structures()
    x_sizes,               # Pixel sizes in x (same shape as labelled_array)
    y_sizes,               # Pixel sizes in y
) -> np.ndarray  # Shape (n_labels,). O(n) via np.bincount.
```

```python
objscale.get_structure_perimeters(
    labelled_array,        # From label_structures()
    nan_mask,              # From label_structures()
    n_labels,              # From label_structures()
    x_sizes,               # Pixel sizes in x (same shape as labelled_array)
    y_sizes,               # Pixel sizes in y
) -> np.ndarray  # Shape (n_labels,). O(n) parallel Numba.
```

```python
objscale.get_structure_height_width(
    labelled_array,        # From label_structures()
    nan_mask,              # From label_structures()
    n_labels,              # From label_structures()
    x_sizes,               # Pixel sizes in x (same shape as labelled_array)
    y_sizes,               # Pixel sizes in y
) -> (heights, widths)    # Each shape (n_labels,)
```

```python
# get_structure_props is a convenience wrapper that accepts a binary array,
# labels internally, and returns filtered results (zero-area labels removed).
objscale.get_structure_props(
    array,                 # Binary array (0s, 1s, nans). Assumes toroidal periodicity.
    x_sizes,               # Pixel sizes in x (same shape as array)
    y_sizes,               # Pixel sizes in y
    structure=...,         # Connectivity (default: 4-connected)
    print_none=False,      # Print message if no structures
) -> (perimeters, areas, heights, widths)
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
objscale.isolate_nth_largest_structure(
    binary_array,    # Binary input array
    n=1,             # Which structure (1=largest, 2=second largest, ...)
    structure=...    # Connectivity (default: 4-connected)
) -> np.ndarray[bool]   # Boolean array with only Nth largest structure True
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

**Caveat**: the returned errors assume the data points are statistically independent. This holds for generic regression tasks, but NOT for scaling functions computed from fractal/multifractal fields — do not use these errors as exponent uncertainties (see [Critical Usage Notes](#no-uncertainty-estimates-are-returned--deliberately)).

```python
objscale.skill_path() -> pathlib.Path   # Path to the SKILL.md bundled with the installed version

objscale.install_agent_skill(
    agent    # Required: 'claude' or 'codex'. Copies the bundled skill to
             # ~/.claude/skills/objscale/ or ~/.codex/skills/objscale/.
             # For other agent frameworks, copy skill_path() manually.
) -> pathlib.Path   # Destination path
```

```python
objscale.set_num_threads(
    n    # Number of threads for parallel computations (Numba)
) -> None
```

```python
objscale.get_coords_of_boundaries(
    array    # 2D binary array
) -> np.ndarray   # Shape (n_boundaries, 2) - indices of boundary pixels (toroidal topology)
```

```python
objscale.get_locations_from_pixel_sizes(
    pixel_sizes_x,    # 2D array of pixel sizes in x
    pixel_sizes_y     # 2D array of pixel sizes in y
) -> (locations_x, locations_y)   # Cumulative location coordinates
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

dim = objscale.ensemble_correlation_dimension(
    arrays, x_sizes=x_sizes, y_sizes=y_sizes
)
```

## References

If using this package, cite **both** papers:

### DeWitt & Garrett (2024) - Size Distributions

**Finite domains cause bias in measured and modeled distributions of cloud sizes.**
DeWitt, T. D. and Garrett, T. J., *Atmos. Chem. Phys.*, 24, 8457–8472, 2024. https://doi.org/10.5194/acp-24-8457-2024

> A significant uncertainty in assessments of the role of clouds in climate is the characterization of the full distribution of their sizes. Order-of-magnitude disagreements exist among observations of key distribution parameters, particularly power law exponents and the range over which they apply. A study by Savre and Craig (2023) suggested that the discrepancies are due in large part to inaccurate fitting methods: they recommended the use of a maximum likelihood estimation technique rather than a linear regression to a logarithmically transformed histogram of cloud sizes. Here, we counter that linear regression is both simpler and equally accurate, provided the simple precaution is followed that bins containing fewer than ~24 counts are omitted from the regression. A much more significant and underappreciated source of error is how to treat clouds that are truncated by the edges of unavoidably finite measurement domains. We offer a simple computational procedure to identify and correct for domain size effects, with potential application to any geometric size distribution of objects, whether physical, ecological, social or mathematical.

### DeWitt et al. (2026) - Fractal Dimensions

**Toward less subjective metrics for quantifying the shape and organization of clouds.**
DeWitt, T. D., Garrett, T. J., and Rees, K. N., *Atmos. Chem. Phys.*, 26, 6951–6971, 2026. https://doi.org/10.5194/acp-26-6951-2026

> As cloud sizes and shapes become better resolved by numerical climate models, objective metrics are required to evaluate whether simulations satisfactorily reflect observations. However, even the most recent cloud classification schemes rely on subjectively defined visual categories that lack any direct connection to the underlying physics. The fractal dimension of cloud fields has been used to provide a more objective footing. But, as we describe here, there are a wide range of largely unrecognized subtleties to such analyses that must be considered prior to obtaining meaningfully quantitative results. Methods are described for calculating two distinct types of fractal dimension: an individual fractal dimension Di representing the roughness of individual cloud edges, and an ensemble fractal dimension De characterizing how cloud fields organize hierarchically across spatial scales. Both have the advantage that they can be linked to physical symmetry principles, but De is argued to be better suited for observational validation of simulated collections of clouds, particularly when it is calculated using a straightforward correlation integral method. A remaining challenge is an observed sensitivity of calculated values of De to subjective choices of the reflectivity threshold used to distinguish clouds from clear skies. We advocate that, in the interests of maximizing objectivity, future work should consider treating cloud ensembles as continuous reflectivity fields rather than collections of discrete objects.
