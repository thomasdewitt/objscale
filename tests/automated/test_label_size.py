#!/usr/bin/env python3
"""Tests for label_size bugs found by codex review."""

import numpy as np
import sys
sys.path.insert(0, '../..')
import objscale


def test_float_truncation():
    """Bug 1: label_size truncates float sizes to int.

    A 2x2 structure with pixel sizes of 1.5 should have area = 2.25 per pixel,
    total area = 9.0. But the output array is dtype=int, so 9.0 -> 9 (ok here),
    but for non-round values like pixel size 1.3: area per pixel = 1.69,
    total = 6.76 which truncates to 6.
    """
    array = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    x_sizes = np.full_like(array, 1.3)
    y_sizes = np.full_like(array, 1.3)

    result = objscale.label_size(array, variable='area', x_sizes=x_sizes, y_sizes=y_sizes)

    # Each pixel area = 1.3 * 1.3 = 1.69, total = 4 * 1.69 = 6.76
    expected_area = 4 * 1.3 * 1.3
    actual = result[1, 1]

    if abs(actual - expected_area) < 0.01:
        print(f"  PASS  float_truncation: area={actual:.4f} (expected {expected_area:.4f})")
        return True
    else:
        print(f"  FAIL  float_truncation: area={actual} (expected {expected_area:.4f}, dtype={result.dtype})")
        return False


def test_nan_connectivity():
    """Bug 2: NaN treated as structure during labeling.

    Two separate 1-pixel structures separated by NaN. Because nan.astype(bool)
    is True, the NaN pixels get labeled as structure, potentially merging
    the two separate structures into one.
    """
    n = np.nan
    array = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, n, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    result = objscale.label_size(array, variable='area')

    # The two pixels should be separate structures, each with area=1
    val_left = result[1, 1]
    val_right = result[1, 3]

    # If NaN merges them, they'd both show area=3 (two 1-pixels + one NaN pixel)
    # Correct: each should be area=1
    if val_left == 1 and val_right == 1:
        print(f"  PASS  nan_connectivity: left={val_left}, right={val_right} (separate structures)")
        return True
    else:
        print(f"  FAIL  nan_connectivity: left={val_left}, right={val_right} (expected both=1, NaN may have merged them)")
        return False


if __name__ == '__main__':
    passed = 0
    failed = 0

    for test in [test_float_truncation, test_nan_connectivity]:
        ok = test()
        passed += ok
        failed += (not ok)

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    sys.exit(0 if failed == 0 else 1)
