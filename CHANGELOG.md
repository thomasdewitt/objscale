# Changelog

Notable changes to objscale. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries before 2.0.0 are reconstructed retrospectively from commit history.

## [2.0.0] - 2026-07-13

### Removed (breaking)

- **All exponent/dimension estimators no longer return uncertainty estimates.** Functions that returned `(value, uncertainty)` now return the point estimate alone; `return_values`/`return_C_l`/`return_counts` tuples drop the uncertainty element as well. The removed uncertainties were 2× the OLS standard error of the log-log regression, which assumes statistically independent points — an assumption that fails for scaling functions of fractal/multifractal fields, making the reported uncertainties badly miscalibrated (demonstrated by bootstrap in ["Too many exponents"](https://thomasddewitt.com/thought-cloud/too-many-exponents/index.html)). Users needing uncertainty should bootstrap across statistically independent images. Affected: `ensemble_correlation_dimension`, `ensemble_sandbox_renyi_dimension`, `ensemble_box_renyi_dimension`, `ensemble_box_dimension`, `ensemble_information_dimension`, `individual_correlation_dimension`, `individual_fractal_dimension`, `finite_array_powerlaw_exponent`. `linear_regression` is unchanged (its errors are valid for independent data).
- `ensemble_coarsening_dimension` removed entirely (disabled since 1.4.0; binary-array coarsening is ill-defined).

### Added

- The agent skill file (`objscale/SKILL.md`) now ships inside the pip package. New helpers: `objscale.skill_path()` returns the bundled skill's path; `objscale.install_agent_skill('claude' | 'codex')` installs it to the corresponding agent skills directory.
- Package-level module docstring, including agent-skill discovery instructions.
- Skill file: new guidance on scaling sanity checks, discretization/finite-domain bias, fitting-range choice, and uncertainty; references to the "Too many exponents" post.
- This changelog.

### Changed

- Citations updated: DeWitt et al. fractal-dimensions paper is now published — DeWitt, Garrett, & Rees (2026), *Toward less subjective metrics for quantifying the shape and organization of clouds*, ACP 26, 6951–6971, doi:10.5194/acp-26-6951-2026. Cite both this and DeWitt & Garrett (2024, doi:10.5194/acp-24-8457-2024) when using the package.
- Skill source-of-truth moved from `agent-skills/objscale/SKILL.md` to `objscale/SKILL.md`.

## [1.4.0] - 2026-06-01

### Added

- Full Rényi dimension family: `ensemble_sandbox_renyi_dimension`, `ensemble_box_renyi_dimension`, `ensemble_information_dimension`; `ensemble_correlation_dimension` and `ensemble_box_dimension` became thin wrappers (q=2 sandbox, q=0 box respectively).
- Parallel numba box-sum kernel (large speedups for box-counting, up to several hundred× on large arrays).
- `individual_fractal_dimension` `method=` strings (six perimeter-vs-length-scale combinations) and `min_length_scale`/`max_length_scale` parameters.

### Changed

- `interior_circles_only` default flipped `True` → `False` on ensemble estimators (empirically less biased; see the "Correlation dimension and domain boundary effects" discussion in "Too many exponents").
- Memory optimizations across the package (bit-identical outputs).

### Deprecated

- `variable='perimeter'` → `'summed perimeter'`; `filled=`/`min_a=`/`max_a=` on `individual_fractal_dimension` → `method=`/`min_length_scale=`/`max_length_scale=`.
- `ensemble_coarsening_dimension` disabled (raises, pointing at alternatives).

## [1.3.0] - 2026-04-03

- Added `filled=` option to individual dimension functions (hole-filling before computing).
- `get_structure_areas`/`get_structure_perimeters`/`get_structure_height_width` now require pre-labeled arrays from `label_structures`, with label-aligned outputs; `get_structure_props` retained as the binary-array convenience wrapper.

## [1.2.0] - 2026-04-03

- O(n) parallel numba kernels for structure areas and perimeters; `label_size` bug fixes; import cleanup.

## [1.1.0] - 2026-04-02

- Binned fitting for `individual_fractal_dimension`; expose per-object perimeter/area values for plotting.

## [1.0.0] - 2026-04-02

- Added `individual_correlation_dimension`; renamed `isolate_largest_structure` → `isolate_nth_largest_structure`. Declared stable.

## [0.x] - 2025-07 – 2026-03

- Initial public releases: size distributions with finite-domain truncation corrections, correlation and box dimensions, object analysis utilities, numba parallelization, docs at readthedocs, Zenodo archival (0.1.7).
