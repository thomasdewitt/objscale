#!/usr/bin/env python3
"""
Automated tests for finite_array_size_distribution truncated/nontruncated counting.

Each test array is 15x15 with manually placed clouds. We verify that the total
number of truncated and nontruncated objects matches the expected count for
all 5 variable options: area, perimeter, nested perimeter, height, width.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
import objscale


def get_total_counts(array, variable):
    """Run finite_array_size_distribution and return (nontruncated, truncated) total counts."""
    _, nontrunc, trunc, _ = objscale.finite_array_size_distribution(
        array, variable=variable, bins=50, bin_logs=True, min_threshold=0.5
    )
    return int(round(nontrunc.sum())), int(round(trunc.sum()))


# fmt: off

# ============================================================================
# Array 1: Two interior clouds, none touching edge
# ============================================================================
ARRAY_1 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_1 = {
    'area':             (2, 0),  # 2 interior clouds
    'summed perimeter':        (2, 0),
    'nested perimeter': (2, 0),  # no holes, same as perimeter
    'height':           (2, 0),
    'width':            (2, 0),
}
# ============================================================================
# Array 1.5: One spanning cld
# ============================================================================
ARRAY_1_5 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_1_5 = {
    'area':             (0, 1),  # 2 interior clouds
    'summed perimeter':        (0, 1),
    'nested perimeter': (0, 1),  # no holes, same as perimeter
    'height':           (0, 1),
    'width':            (0, 1),
}

# ============================================================================
# Array 2: One cloud touching left edge, one interior cloud
# ============================================================================
ARRAY_2 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_2 = {
    'area':             (1, 1),  # 1 interior 3x3, 1 edge-touching 2x2
    'summed perimeter':        (1, 1),
    'nested perimeter': (1, 1),  # no holes
    'height':           (1, 1),
    'width':            (1, 1),
}

# ============================================================================
# Array 3: Two clouds touching different edges, no interior clouds
# ============================================================================
ARRAY_3 = np.array([
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_3 = {
    'area':             (0, 2),  # both touch edges
    'summed perimeter':        (0, 2),
    'nested perimeter': (0, 2),  # This is actually ambiguous - could be either considered a edge hole or a concavity. Assume concavity 
    'height':           (0, 2),
    'width':            (0, 2),
}

# ============================================================================
# Array 4: Interior donut (one object with a hole) + small interior cloud
#          inside hole
# ============================================================================
ARRAY_4 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,1,1,0,1,0,1,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_4 = {
    'area':             (2, 0),  # donut (1 object) + 1px cloud inside hole
    'summed perimeter':        (2, 0),
    'nested perimeter': (3, 0),  # donut outer + donut hole + 1px cloud = 3
    'height':           (2, 0),
    'width':            (2, 0),
}

# ============================================================================
# Array 5: 1 cloud touching edge w two holes, one touching one not touching
# Main cloud touches left edge; has edge-touching hole (rows 4-6, cols 0)
# and interior hole at (6,6)
# ============================================================================
ARRAY_5 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,1,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_5 = {
    'area':             (0, 1),  # 1 edge-touching cloud
    'summed perimeter':        (0, 1),
    'nested perimeter': (1, 1),  # interior hole nontrunc; edge hole + outer boundary trunc
    'height':           (0, 1),
    'width':            (0, 1),
}

# ============================================================================
# Array 6: 1 cloud touching edge w two holes, one touching one not touching,
#          hole not touching edge has interior cloud
# Same as Array 5 but with bigger interior hole containing a 1px cloud
# ============================================================================
ARRAY_6 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [1,1,0,0,0,1,0,0,0,0,1,0,0,0,0],
    [0,1,0,0,0,1,0,0,1,0,1,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,0,1,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_6 = {
    'area':             (1, 1),  # 1 edge-touching cloud (1px inside hole is separate obj)
    'summed perimeter':        (1, 1),
    'nested perimeter': (2, 1),  # interior hole + 1px cloud nontrunc; edge hole + outer trunc
    'height':           (1, 1),
    'width':            (1, 1),
}


# ============================================================================
# Array 7: Double nested donut w 1 edge touching cloud
# ============================================================================
ARRAY_7 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,1,1,1,1,1,0,1,0],
    [1,1,0,0,0,1,0,1,0,0,0,1,0,1,0],
    [0,1,0,0,0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,0,0,1,0,1,0,0,0,1,0,1,0],
    [1,1,1,0,1,1,0,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_7 = {
    'area':             (3, 1),  # 1 edge-touching cloud (1px inside hole is separate obj)
    'summed perimeter':        (3, 1),
    'nested perimeter': (5, 1),  # interior hole + 1px cloud nontrunc; edge hole + outer trunc
    'height':           (3, 1),
    'width':            (3, 1),
}

# ============================================================================
# Array 8: Double nested donut touching edge + 1 edge touching cloud
# ============================================================================
ARRAY_8 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,1,1,1,1,1,0,1,0],
    [1,1,0,0,0,1,0,1,0,0,0,1,0,1,0],
    [0,1,0,0,0,1,0,1,0,1,0,1,0,1,0],
    [0,1,0,0,0,1,0,1,0,0,0,1,0,1,0],
    [1,1,1,0,1,1,0,1,1,1,1,1,0,1,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_8 = {
    'area':             (2, 2), 
    'summed perimeter':        (2, 2),
    'nested perimeter': (4, 2), 
    'height':           (2, 2),
    'width':            (2, 2),
}

# fmt: on


# ============================================================================
# Collect arrays and expected values
# ============================================================================
CASES = {
    'ARRAY_1': (ARRAY_1, EXPECTED_1),
    'ARRAY_1.5': (ARRAY_1_5, EXPECTED_1_5),
    'ARRAY_2': (ARRAY_2, EXPECTED_2),
    'ARRAY_3': (ARRAY_3, EXPECTED_3),
    'ARRAY_4': (ARRAY_4, EXPECTED_4),
    'ARRAY_5': (ARRAY_5, EXPECTED_5),
    'ARRAY_6': (ARRAY_6, EXPECTED_6),
    'ARRAY_7': (ARRAY_7, EXPECTED_7),
    'ARRAY_8': (ARRAY_8, EXPECTED_8),
}

VARIABLES = ['area', 'summed perimeter', 'nested perimeter', 'height', 'width']


def run_test(case_name, variable):
    """Test a single array/variable combination against the expected key."""
    array, expected = CASES[case_name]
    expected_nontrunc, expected_trunc = expected[variable]
    nontrunc, trunc = get_total_counts(array, variable)
    assert nontrunc == expected_nontrunc, (
        f"{case_name} / {variable}: expected {expected_nontrunc} nontruncated, got {nontrunc}"
    )
    assert trunc == expected_trunc, (
        f"{case_name} / {variable}: expected {expected_trunc} truncated, got {trunc}"
    )


# Make the 45-case matrix collectable by pytest (script-mode runner below is
# unchanged).
try:
    import pytest

    @pytest.mark.parametrize('case_name', list(CASES))
    @pytest.mark.parametrize('variable', VARIABLES)
    def test_finite_size_distribution_matrix(case_name, variable):
        run_test(case_name, variable)
except ImportError:
    pass


# ============================================================================
# Regression tests for the 2.0.0 review fixes (1, 6, 7) and shape contracts
# ============================================================================

def test_fix1_default_bins_length_units():
    """Fix 1: variable-aware default bins keep sub-unity-pixel objects.

    Codex repro: 50x50 grid, dx=dy=0.01, a 40-pixel interior line. With the old
    [min_threshold, domain-area] default, length-unit variables were binned over
    a range whose upper edge (0.25) was below min_threshold (10) and the line was
    silently discarded. With variable-aware default bins it is counted.
    """
    grid = np.zeros((50, 50), dtype=np.float32)
    grid[25, 5:45] = 1  # interior horizontal line, 40 px long, does not touch edge
    dx = np.full((50, 50), 0.01, dtype=np.float64)
    dy = np.full((50, 50), 0.01, dtype=np.float64)
    for variable in ('summed perimeter', 'width', 'area'):
        _, nontrunc, trunc, _ = objscale.finite_array_size_distribution(
            grid, variable=variable, x_sizes=dx, y_sizes=dy, bins=50)
        total = int(round(nontrunc.sum() + trunc.sum()))
        assert total == 1, (
            f"{variable}: expected the interior line to be counted once with "
            f"default bins, got total={total}"
        )


def test_fix6_finite_accepts_list_bins():
    """Fix 6: finite_array_size_distribution accepts array-like (list) bins."""
    arr = np.zeros((30, 30), dtype=np.float32)
    arr[5:8, 5:8] = 1
    edges = list(np.linspace(0.0, 3.0, 21))  # log10 edges (bin_logs default True)
    out = objscale.finite_array_size_distribution(arr, variable='area', bins=edges)
    assert len(out) == 4  # (bin_middles, nontrunc, trunc, truncation_index)


def test_fix6_array_size_distribution_log_midpoints():
    """Fix 6: per-bin midpoints are edges-based (correct for non-uniform edges)."""
    arr = np.zeros((30, 30), dtype=np.float32)
    arr[5:8, 5:8] = 1
    edges = np.geomspace(1.0, 100.0, 11)  # non-uniform (log-spaced) linear edges
    mids, _ = objscale.array_size_distribution(
        arr, variable='area', bins=edges, bin_logs=False)
    expected = 0.5 * (edges[:-1] + edges[1:])
    assert np.allclose(mids, expected), (
        f"midpoints not edges-based: {mids} vs {expected}"
    )


def test_fix7_truncation_index_no_truncation_sentinel():
    """Fix 7: with no truncated bin, truncation_index is one past the counts."""
    _, nontrunc, trunc, trunc_idx = objscale.finite_array_size_distribution(
        ARRAY_1, variable='area', bins=50, min_threshold=0.5)
    assert len(nontrunc) == 50
    assert trunc_idx == len(nontrunc), (
        f"expected sentinel {len(nontrunc)}, got {trunc_idx}"
    )


def test_shape_contract_finite_powerlaw_return_counts():
    """2.0.0 contract: finite_array_powerlaw_exponent(return_counts=True) -> (float, (arr, arr))."""
    arr = np.zeros((30, 30), dtype=np.float32)
    arr[5:8, 5:8] = 1
    out = objscale.finite_array_powerlaw_exponent(
        [arr], 'area', bins=20, return_counts=True)
    assert isinstance(out, tuple) and len(out) == 2
    exponent, counts = out
    assert isinstance(exponent, float)
    assert isinstance(counts, tuple) and len(counts) == 2
    assert isinstance(counts[0], np.ndarray) and isinstance(counts[1], np.ndarray)


REGRESSION_TESTS = [
    test_fix1_default_bins_length_units,
    test_fix6_finite_accepts_list_bins,
    test_fix6_array_size_distribution_log_midpoints,
    test_fix7_truncation_index_no_truncation_sentinel,
    test_shape_contract_finite_powerlaw_return_counts,
]


if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []
    for case_name in CASES:
        for variable in VARIABLES:
            label = f"{case_name} / {variable}"
            try:
                run_test(case_name, variable)
                # print(f"  PASS: {label}")
                passed += 1
            except Exception as e:
                print(e)
                failed += 1

    for fn in REGRESSION_TESTS:
        try:
            fn()
            print(f"  PASS: {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {fn.__name__} -- {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
