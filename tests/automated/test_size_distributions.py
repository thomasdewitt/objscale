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
    'perimeter':        (2, 0),
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
    'perimeter':        (0, 1),
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
    'perimeter':        (1, 1),
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
    'perimeter':        (0, 2),
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
    'perimeter':        (2, 0),
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
    'perimeter':        (0, 1),
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
    'perimeter':        (1, 1),
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
    'perimeter':        (3, 1),
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
    'perimeter':        (2, 2),
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

VARIABLES = ['area', 'perimeter', 'nested perimeter', 'height', 'width']


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
                # print(f"  PASS: {label}")
                passed += 1
            except Exception as e:
                print(e)

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  - {e}")
