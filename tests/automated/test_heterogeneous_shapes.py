"""
Tests for the x_sizes/y_sizes contract and heterogeneous-shape handling.

Contract (see finite_array_* / individual_fractal_dimension docstrings):

  - x_sizes/y_sizes may be a LIST: one grid per array, each matching its
    array's shape (else raise).
  - a SINGLE ndarray: every array must match its shape (else raise).
  - None: unit pixels, no shared grid built; arrays of DIFFERENT shapes are
    pooled correctly (the size-distribution / individual-fractal-dimension
    functions aggregate per-object quantities, so shape need not be shared).

Functions that genuinely require a shared coordinate space (the ensemble
sandbox/correlation dimension) must RAISE a clear error on mismatched shapes
rather than silently misaligning.

Regression origin: a MODIS granule pipeline (Oppong, MODIS-Cloud-Geometry)
had to NaN-/zero-pad every granule to a common shape to work around
finite_array_powerlaw_exponent misaligning an internal boolean
(`labels_flat > 0`) against a pixel-area grid built from arrays[0].shape.
"""
import numpy as np
import pytest
import objscale
from objscale._object_analysis import label_structures, get_structure_areas


def _blobs(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < 0.3).astype(float)


# ---------------------------------------------------------------------------
# 1. Heterogeneous shapes + None must WORK for pooled size distributions
# ---------------------------------------------------------------------------

def test_finite_powerlaw_heterogeneous_shapes_no_crash():
    """Different-shaped arrays with x_sizes=None must not raise (was IndexError)."""
    a = _blobs((60, 90), 0)
    b = _blobs((50, 70), 1)  # different total size -> old code raised IndexError
    exp = objscale.finite_array_powerlaw_exponent([a, b], "area", bins=20)
    assert np.isfinite(exp)


def test_finite_sizedist_pooling_is_additive_across_shapes():
    """Pooled counts over differently-shaped arrays == sum of per-array counts.

    This is the load-bearing correctness invariant: finite_array_size_distribution
    pools object counts linearly across arrays, so with identical explicit bins
    the two-array result must equal the elementwise sum of the one-array results,
    even when the arrays have different shapes.
    """
    a = _blobs((60, 90), 2)
    b = _blobs((50, 70), 3)
    edges = np.linspace(0.0, 3.0, 21)  # log10(area) bin edges

    _, nt_ab, t_ab, _ = objscale.finite_array_size_distribution([a, b], "area", bins=edges)
    _, nt_a, t_a, _ = objscale.finite_array_size_distribution([a], "area", bins=edges)
    _, nt_b, t_b, _ = objscale.finite_array_size_distribution([b], "area", bins=edges)

    np.testing.assert_allclose(nt_ab, nt_a + nt_b)
    np.testing.assert_allclose(t_ab, t_a + t_b)


def test_individual_fractal_dimension_heterogeneous_shapes():
    """individual_fractal_dimension pools per-object points; shapes may differ."""
    a = _blobs((80, 120), 4)
    b = _blobs((70, 60), 5)
    dim = objscale.individual_fractal_dimension([a, b], bins=10)
    assert np.isfinite(dim)


# ---------------------------------------------------------------------------
# 2. None area weights == unit-grid area (the pixel_areas fast-path)
# ---------------------------------------------------------------------------

def test_get_structure_areas_none_equals_unit_grid():
    arr = _blobs((40, 50), 6)
    lab, nm, nl = label_structures(arr, wrap=None)
    ones = np.ones(lab.shape, dtype=np.float32)
    a_none = get_structure_areas(lab, nm, nl, None, None)
    a_ones = get_structure_areas(lab, nm, nl, ones, ones)
    np.testing.assert_allclose(a_none, a_ones)


def test_mixed_none_pixel_sizes_raises():
    """Exactly one of x_sizes/y_sizes being None is ambiguous -> must raise.

    Either pass both (a shared unit assumption) or neither. This avoids the
    footgun where the area path and the auto-bin range disagree about whether
    the provided axis is used.
    """
    arr = _blobs((30, 40), 21)
    y2 = np.full(arr.shape, 2.0, dtype=np.float32)

    # low-level entry point
    lab, nm, nl = label_structures(arr, wrap=None)
    y2_lab = np.full(lab.shape, 2.0, dtype=np.float32)
    with pytest.raises(ValueError):
        get_structure_areas(lab, nm, nl, None, y2_lab)
    with pytest.raises(ValueError):
        get_structure_areas(lab, nm, nl, y2_lab, None)

    # public size-distribution entry point
    with pytest.raises(ValueError):
        objscale.finite_array_powerlaw_exponent([arr], "area", x_sizes=None, y_sizes=y2, bins=20)


# ---------------------------------------------------------------------------
# 3. Shape-contract violations must RAISE
# ---------------------------------------------------------------------------

def test_single_xsizes_mismatch_raises():
    a = _blobs((60, 90), 7)
    b = _blobs((50, 70), 8)
    x = np.ones((60, 90), dtype=np.float32)  # matches a, not b
    with pytest.raises(ValueError):
        objscale.finite_array_powerlaw_exponent([a, b], "area", x_sizes=x, y_sizes=x, bins=20)


def test_list_xsizes_wrong_length_raises():
    a = _blobs((60, 90), 9)
    b = _blobs((60, 90), 10)
    x = [np.ones((60, 90), dtype=np.float32)]  # only one grid for two arrays
    with pytest.raises(ValueError):
        objscale.finite_array_powerlaw_exponent([a, b], "area", x_sizes=x, y_sizes=x, bins=20)


def test_list_xsizes_element_shape_mismatch_raises():
    a = _blobs((60, 90), 11)
    b = _blobs((50, 70), 12)
    x = [np.ones((60, 90), np.float32), np.ones((60, 90), np.float32)]  # x[1] != b.shape
    with pytest.raises(ValueError):
        objscale.finite_array_powerlaw_exponent([a, b], "area", x_sizes=x, y_sizes=x, bins=20)


# ---------------------------------------------------------------------------
# 4. Ensemble dimension requires a shared shape -> clear raise
# ---------------------------------------------------------------------------

def test_ensemble_correlation_dimension_mismatched_shapes_raises():
    a = _blobs((120, 120), 13)
    b = _blobs((100, 140), 14)
    with pytest.raises(ValueError, match=r"(?i)shape"):
        objscale.ensemble_correlation_dimension([a, b], point_reduction_factor=2, nbins=30)
