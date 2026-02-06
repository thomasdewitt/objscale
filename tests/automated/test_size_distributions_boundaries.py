#!/usr/bin/env python3
"""
Automated tests for finite_array_size_distribution with irregular NaN boundaries.

Each test array is 15x15 with NaN regions defining non-rectangular valid domains.
All clouds are simple 2x2 blocks (no holes/nesting). We test whether clouds
adjacent to NaN boundaries are correctly classified as truncated vs nontruncated.

Key behavior: only NaN pixels connected to the domain edge (via other NaN) count
as "border NaN". Isolated interior NaN pixels are treated like 0 (background),
so clouds touching them are NOT considered truncated.
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
import objscale

n = np.nan


def get_total_counts(array, variable):
    """Run finite_array_size_distribution and return (nontruncated, truncated) total counts."""
    _, nontrunc, trunc, _ = objscale.finite_array_size_distribution(
        array, variable=variable, bins=50, bin_logs=True, min_threshold=0.5
    )
    return int(round(nontrunc.sum())), int(round(trunc.sum()))


# fmt: off

# ============================================================================
# Array 1: L-shaped valid region (upper-right quadrant is NaN)
# Cloud A: 2x2 at (9,2) — interior of valid region → nontrunc
# Cloud B: 2x2 at (2,5) — adjacent to NaN boundary on right → trunc
# Cloud C: 2x2 at (9,9) — interior of lower region → nontrunc
# ============================================================================
ARRAY_1 = np.array([
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_1 = {
    'area':             (2, 0),
    'perimeter':        (2, 0),
    'height':           (2, 0),
    'width':            (2, 0),
}

# ============================================================================
# Array 2: NaN corridor splits domain horizontally (rows 6-7 all NaN)
# Cloud A: 2x2 at (2,6) — upper half, interior → nontrunc
# Cloud B: 2x2 at (4,6) — upper half, adjacent to NaN strip → trunc
# Cloud C: 2x2 at (10,6) — lower half, interior → nontrunc
# ============================================================================
ARRAY_2 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [n,n,n,n,n,n,n,n,n,n,n,n,n,n,n],
    [n,n,n,n,n,n,n,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_2 = {
    'area':             (1, 1),  # upper cloud touches NaN strip, lower is interior
    'perimeter':        (1, 1),
    'height':           (1, 1),
    'width':            (1, 1),
}

# ============================================================================
# Array 3: Isolated interior NaN island (NOT connected to any edge)
# NaN block at rows 6-8, cols 6-8 surrounded by valid data
# Cloud A: 2x2 at (5,5) — adjacent to NaN island → should be NONTRUNC
#          (isolated NaN is not border NaN!)
# Cloud B: 2x2 at (0,0) — touches domain edge → trunc
# Cloud C: 2x2 at (11,11) — far interior → nontrunc
# ============================================================================
ARRAY_3 = np.array([
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,n,n,n,0,0,0,0,0,0],
    [0,0,0,0,0,0,n,n,n,0,0,0,0,0,0],
    [0,0,0,0,0,0,n,n,n,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_3 = {
    'area':             (2, 1),  # NaN-adjacent cloud is NOT truncated; edge cloud is
    'perimeter':        (2, 1),
    'height':           (2, 1),
    'width':            (2, 1),
}

# ============================================================================
# Array 4: NaN notch cutting in from left edge (rows 6-7, cols 0-4)
# Cloud A: 2x2 at (6,5) — adjacent to notch tip → trunc
# Cloud B: 2x2 at (6,10) — interior, same rows → nontrunc
# Cloud C: 2x2 at (2,2) — interior, away from notch → nontrunc
# ============================================================================
ARRAY_4 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [n,n,n,n,n,1,1,0,0,0,1,1,0,0,0],
    [n,n,n,n,n,1,1,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_4 = {
    'area':             (2, 1),  # notch-adjacent cloud trunc; other two interior
    'perimeter':        (2, 1),
    'height':           (2, 1),
    'width':            (2, 1),
}

# ============================================================================
# Array 5: Diagonal NaN boundary (staircase from top-right to bottom-left)
# Valid region is lower-left triangle
# Cloud A: 2x2 at (10,2) — deep interior → nontrunc
# Cloud B: 2x2 at (5,5) — adjacent to diagonal NaN → trunc
# Cloud C: 2x2 at (12,0) — touches left domain edge → trunc
# ============================================================================
ARRAY_5 = np.array([
    [0,0,n,n,n,n,n,n,n,n,n,n,n,n,n],
    [0,0,0,n,n,n,n,n,n,n,n,n,n,n,n],
    [0,0,0,0,n,n,n,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,n,n,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,n,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,1,1,n,n,n,n,n,n,n,n],
    [0,0,0,0,0,1,1,0,n,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,0,0,n,n,n,n,n,n],
    [0,0,0,0,0,0,0,0,0,0,n,n,n,n,n],
    [0,0,0,0,0,0,0,0,0,0,0,n,n,n,n],
    [0,0,1,1,0,0,0,0,0,0,0,0,n,n,n],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,n,n],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,n],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_5 = {
    'area':             (1, 2),  # interior cloud nontrunc; diagonal-adjacent + left-edge trunc
    'perimeter':        (1, 2),
    'height':           (1, 2),
    'width':            (1, 2),
}

# ============================================================================
# Array 6: NaN channel from top edge to interior (does NOT reach bottom)
# Channel at col 7, rows 0-8. Clouds on either side.
# Cloud A: 2x2 at (3,5) — adjacent to channel (col 6 touches NaN col 7) → trunc
# Cloud B: 2x2 at (3,9) — NOT adjacent (col 8 is 0, separates from NaN col 7) → nontrunc
# Cloud C: 2x2 at (11,6) — below channel, interior → nontrunc
# ============================================================================
ARRAY_6 = np.array([
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,n,0,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,n,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,n,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_6 = {
    'area':             (2, 1),  # Cloud A trunc (touches channel); B + C nontrunc
    'perimeter':        (2, 1),
    'height':           (2, 1),
    'width':            (2, 1),
}

# ============================================================================
# Array 7: Two isolated interior NaN pixels (not connected to edge)
# Cloud A: 2x2 at (3,3) — touching isolated NaN at (3,5) → NONTRUNC
# Cloud B: 2x2 at (7,9) — touching isolated NaN at (7,11) → NONTRUNC
# Cloud C: 2x2 at (12,12) — interior, no nearby NaN → nontrunc
# All three should be nontruncated since isolated NaN ≠ border NaN
# ============================================================================
ARRAY_7 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,n,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,n,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)
#                          (nontrunc, trunc)
EXPECTED_7 = {
    'area':             (3, 0),  # isolated interior NaN does NOT make clouds truncated
    'perimeter':        (3, 0),
    'height':           (3, 0),
    'width':            (3, 0),
}

# fmt: on


# ============================================================================
# Collect arrays and expected values
# ============================================================================
CASES = {
    'ARRAY_1': (ARRAY_1, EXPECTED_1),
    'ARRAY_2': (ARRAY_2, EXPECTED_2),
    'ARRAY_3': (ARRAY_3, EXPECTED_3),
    'ARRAY_4': (ARRAY_4, EXPECTED_4),
    'ARRAY_5': (ARRAY_5, EXPECTED_5),
    'ARRAY_6': (ARRAY_6, EXPECTED_6),
    'ARRAY_7': (ARRAY_7, EXPECTED_7),
}

VARIABLES = ['area', 'perimeter', 'height', 'width']


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


if __name__ == '__main__':
    passed = 0
    failed = 0
    errors = []
    for case_name in CASES:
        for variable in VARIABLES:
            label = f"{case_name} / {variable}"
            try:
                run_test(case_name, variable)
                print(f"  PASS: {label}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL: {label} -- {e}")
                errors.append(label)
                failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
