#!/usr/bin/env python3
"""
Automated tests for periodic boundary behavior in get_structure_*.

get_structure_areas, get_structure_perimeters, get_structure_height_width all
assume toroidal (fully periodic) boundary conditions. For non-periodic domains,
the caller pads edges with 0 or np.nan.

These tests verify:
  1. Fully periodic arrays (no padding) — structures wrap around edges
  2. Non-periodic arrays (nan-padded) — structures stop at padded edges
  3. array_size_distribution with wrap parameter — handles padding internally
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
import objscale

n = np.nan

# fmt: off

# ============================================================================
# Array 1: Structure spanning left-right boundary (fully periodic)
#   Two 1x2 blocks: cols 0-1 and cols 8-9, row 4.
#   Periodic: they form one structure (merged across left-right boundary).
#   area=4, perimeter=10 (top=4, bottom=4, gap at col 2=1, gap at col 7=1)
#   height=1, width=4
# ============================================================================
ARRAY_1 = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)

EXPECTED_1 = {
    'n_structures': 1,
    'areas': [4.0],
    'perimeters': [10.0],
    'height': [1.0],
    'width': [4.0],
}


# ============================================================================
# Array 2: Structure spanning top-bottom boundary (fully periodic)
#   Two 2x1 blocks: rows 0-1 and rows 8-9, col 5.
#   Periodic: one structure, area=4, perimeter=10
#   height=4, width=1
# ============================================================================
ARRAY_2 = np.array([
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
], dtype=np.float32)

EXPECTED_2 = {
    'n_structures': 1,
    'areas': [4.0],
    'perimeters': [10.0],
    'height': [4.0],
    'width': [1.0],
}


# ============================================================================
# Array 3: Corner structure wrapping both directions (fully periodic)
#   Four single pixels at four corners.
#   Periodic: all merge into one 2x2 block.
#   area=4, perimeter=8, height=2, width=2
# ============================================================================
ARRAY_3 = np.array([
    [1,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,1],
], dtype=np.float32)

EXPECTED_3 = {
    'n_structures': 1,
    'areas': [4.0],
    'perimeters': [8.0],
    'height': [2.0],
    'width': [2.0],
}


# ============================================================================
# Array 4: Wrapping structure + interior structure (fully periodic)
#   Edge pixels (3,0) and (3,9) merge into one structure via wrap.
#   Interior 2x2 at (6,4) is separate.
#   2 structures: areas [2, 4], perimeters [6, 8]
# ============================================================================
ARRAY_4 = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)

EXPECTED_4 = {
    'n_structures': 2,
    'areas': sorted([2.0, 4.0]),
    'perimeters': sorted([6.0, 8.0]),
}


# ============================================================================
# Array 5: Full row (fully periodic)
#   Row 2 all 1s. One structure, area=10.
#   Periodic: no edges on left/right (wraps to self).
#   perimeter = top(10) + bottom(10) = 20
#   height=1, width=10
# ============================================================================
ARRAY_5 = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)

EXPECTED_5 = {
    'n_structures': 1,
    'areas': [10.0],
    'perimeters': [20.0],
    'height': [1.0],
    'width': [10.0],
}


# ============================================================================
# Array 6: Non-periodic (nan-padded) — same physical structure as Array 1
#   The nan padding breaks periodicity, so left and right blocks are separate.
#   Each 1x2 block: area=2, perimeter=6 (all 4 sides exposed)
# ============================================================================
ARRAY_6 = np.array([
    [n,n,n,n,n,n,n,n,n,n,n,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,1,1,0,0,0,0,0,0,1,1,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,n,n,n,n,n,n,n,n,n,n,n],
], dtype=np.float32)

EXPECTED_6 = {
    'n_structures': 2,
    'areas': sorted([2.0, 2.0]),
    # NaN edges don't count as perimeter. Each block's outer edge is nan,
    # so only the 0-facing edges contribute: top=2, bottom=2, inner gap=1 → 5 each
    'perimeters': sorted([5.0, 5.0]),
}


# ============================================================================
# Array 7: Non-periodic (nan-padded) — corner pixels are separate
#   With nan padding, no wrapping occurs. 4 separate single-pixel structures.
#   Each: area=1, perimeter=4
# ============================================================================
ARRAY_7 = np.array([
    [n,n,n,n,n,n,n,n,n,n,n,n],
    [n,1,0,0,0,0,0,0,0,0,1,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,0,0,0,0,0,0,0,0,0,0,n],
    [n,1,0,0,0,0,0,0,0,0,1,n],
    [n,n,n,n,n,n,n,n,n,n,n,n],
], dtype=np.float32)

EXPECTED_7 = {
    'n_structures': 4,
    'areas': sorted([1.0, 1.0, 1.0, 1.0]),
    # NaN neighbors don't count as perimeter, only 0 neighbors do.
    # Corner pixels: e.g. (1,1) has nan above, nan left, 0 right, 0 below → perim=2
    'perimeters': sorted([2.0, 2.0, 2.0, 2.0]),
}


# ============================================================================
# Array 8: array_size_distribution with wrap parameter
#   Same base array as Array 1, but called through array_size_distribution
#   which handles padding internally based on wrap.
# ============================================================================
ARRAY_8_BASE = np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
], dtype=np.float32)

# wrap=None: non-periodic → 2 separate structures
# wrap='sides': periodic left-right → 1 merged structure
# wrap='both': fully periodic → 1 merged structure
EXPECTED_8_NOWRAP_COUNT = 2
EXPECTED_8_SIDES_COUNT = 1
EXPECTED_8_BOTH_COUNT = 1

# fmt: on


# =============================================================================
# Fix 4: nested perimeter must respect wrap
# =============================================================================

def _boundary_perims(array, wrap):
    from objscale._object_analysis import get_every_boundary_perimeter
    xs = np.ones_like(array, dtype=np.float32)
    ys = np.ones_like(array, dtype=np.float32)
    return sorted(float(p) for p in
                  get_every_boundary_perimeter(array, xs, ys, wrap=wrap))


def test_nested_perimeter_seam_merges():
    """(a) Two pixels adjacent across a periodic seam form ONE boundary.

    With wrapping the two edge pixels merge into a single 1x2 bar whose
    boundary perimeter is 6; without wrapping they are two separate boundaries.
    """
    arr = np.zeros((5, 4), dtype=np.float32)
    arr[2, 0] = 1
    arr[2, 3] = 1
    assert _boundary_perims(arr, 'both') == [6.0]
    assert _boundary_perims(arr, 'sides') == [6.0]
    # wrap=None: no seam merge -> two separate boundaries (previous behavior)
    assert _boundary_perims(arr, None) == [3.0, 3.0]


def test_nested_perimeter_roll_invariance():
    """(b) For a fully periodic array, sorted nested perimeters are roll-invariant."""
    ring = np.zeros((10, 10), dtype=np.float32)
    ring[1:6, 1:6] = 1
    ring[2:5, 2:5] = 0  # 5x5 ring with 3x3 hole
    base = _boundary_perims(ring, 'both')
    for dr, dc in [(0, 0), (3, 7), (5, 5), (-2, 4), (9, 9), (4, 0), (0, 6)]:
        rolled = np.roll(np.roll(ring, dr, axis=0), dc, axis=1)
        assert _boundary_perims(rolled, 'both') == base, (
            f"roll ({dr},{dc}) changed perimeters: {base} -> "
            f"{_boundary_perims(rolled, 'both')}"
        )


def test_nested_perimeter_donut_crosses_seam():
    """(c) A donut crossing the seam gives 2 boundaries with the ring values."""
    ring = np.zeros((10, 10), dtype=np.float32)
    ring[1:6, 1:6] = 1
    ring[2:5, 2:5] = 0
    # Roll so the ring straddles both seams
    rolled = np.roll(np.roll(ring, -3, axis=0), -3, axis=1)
    assert _boundary_perims(rolled, 'both') == [12.0, 20.0]


def test_nested_perimeter_wrap_none_matches_size_distribution():
    """(d) array_size_distribution forwards wrap to nested perimeter."""
    arr = np.zeros((5, 4), dtype=np.float32)
    arr[2, 0] = 1
    arr[2, 3] = 1
    # wrap=None -> 2 boundaries; wrap='both' -> 1 boundary
    _, counts_none = objscale.array_size_distribution(
        arr, variable='nested perimeter', bins=50, wrap=None)
    _, counts_both = objscale.array_size_distribution(
        arr, variable='nested perimeter', bins=50, wrap='both')
    assert int(round(counts_none.sum())) == 2
    assert int(round(counts_both.sum())) == 1


NESTED_WRAP_TESTS = [
    test_nested_perimeter_seam_merges,
    test_nested_perimeter_roll_invariance,
    test_nested_perimeter_donut_crosses_seam,
    test_nested_perimeter_wrap_none_matches_size_distribution,
]


# =============================================================================
# Test runner
# =============================================================================

def check(label, actual, expected):
    """Check actual vs expected, return True if pass."""
    if isinstance(expected, list):
        actual_sorted = sorted([float(x) for x in actual])
        if actual_sorted != expected:
            print(f"  FAIL  {label}: expected {expected}, got {actual_sorted}")
            return False
    elif isinstance(expected, (int, float)):
        if actual != expected:
            print(f"  FAIL  {label}: expected {expected}, got {actual}")
            return False
    print(f"  PASS  {label}")
    return True


def _run_array_props(name, array, expected):
    """Test get_structure_areas/perimeters/height_width on one array."""
    xs = np.ones_like(array, dtype=np.float32)
    ys = np.ones_like(array, dtype=np.float32)
    passed = 0
    failed = 0

    lab, nm, nl = objscale.label_structures(array)
    if lab is None:
        a = np.array([], dtype=np.float32)
        p = np.array([], dtype=np.float32)
    else:
        a_all = objscale.get_structure_areas(lab, nm, nl, xs, ys)
        p_all = objscale.get_structure_perimeters(lab, nm, nl, xs, ys)
        valid = a_all > 0
        a = a_all[valid]
        p = p_all[valid]

    if 'n_structures' in expected:
        ok = check(f"{name}/n_structures(area)", len(a), expected['n_structures'])
        passed += ok; failed += (not ok)
        ok = check(f"{name}/n_structures(perim)", len(p), expected['n_structures'])
        passed += ok; failed += (not ok)

    if 'areas' in expected:
        ok = check(f"{name}/areas", a, expected['areas'])
        passed += ok; failed += (not ok)

    if 'perimeters' in expected:
        ok = check(f"{name}/perimeters", p, expected['perimeters'])
        passed += ok; failed += (not ok)

    if 'height' in expected or 'width' in expected:
        if lab is None:
            h = np.array([], dtype=np.float32)
            w = np.array([], dtype=np.float32)
        else:
            h_all, w_all = objscale.get_structure_height_width(lab, nm, nl, xs, ys)
            h = h_all[valid]
            w = w_all[valid]
        if 'height' in expected:
            ok = check(f"{name}/height", h, expected['height'])
            passed += ok; failed += (not ok)
        if 'width' in expected:
            ok = check(f"{name}/width", w, expected['width'])
            passed += ok; failed += (not ok)

    return passed, failed


def _run_size_distribution(name, array, wrap, expected_n):
    """Test array_size_distribution returns correct count with wrapping."""
    _, counts = objscale.array_size_distribution(
        array, variable='area', bins=50, wrap=wrap,
    )
    total = int(round(counts.sum()))
    wrap_label = wrap if wrap else 'nowrap'
    ok = check(f"{name}/{wrap_label}/size_dist_count", total, expected_n)
    return int(ok), int(not ok)


def run_all_tests():
    total_passed = 0
    total_failed = 0

    # Fully periodic tests
    tests = [
        ('ARRAY_1_periodic', ARRAY_1, EXPECTED_1),
        ('ARRAY_2_periodic', ARRAY_2, EXPECTED_2),
        ('ARRAY_3_periodic', ARRAY_3, EXPECTED_3),
        ('ARRAY_4_periodic', ARRAY_4, EXPECTED_4),
        ('ARRAY_5_periodic', ARRAY_5, EXPECTED_5),
        # Non-periodic (nan-padded) tests
        ('ARRAY_6_nanpadded', ARRAY_6, EXPECTED_6),
        ('ARRAY_7_nanpadded', ARRAY_7, EXPECTED_7),
    ]

    for name, array, expected in tests:
        p, f = _run_array_props(name, array, expected)
        total_passed += p
        total_failed += f

    # array_size_distribution wrap tests
    sd_tests = [
        ('ARRAY_8', ARRAY_8_BASE, None, EXPECTED_8_NOWRAP_COUNT),
        ('ARRAY_8', ARRAY_8_BASE, 'sides', EXPECTED_8_SIDES_COUNT),
        ('ARRAY_8', ARRAY_8_BASE, 'both', EXPECTED_8_BOTH_COUNT),
    ]
    for name, array, wrap, expected_n in sd_tests:
        p, f = _run_size_distribution(name, array, wrap, expected_n)
        total_passed += p
        total_failed += f

    # Nested-perimeter wrap tests (Fix 4)
    for fn in NESTED_WRAP_TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            total_passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            total_failed += 1

    print(f"\n{total_passed} passed, {total_failed} failed out of {total_passed + total_failed} tests")
    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
