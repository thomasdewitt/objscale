#!/usr/bin/env python3
"""
Automated tests for objscale using seed-based recursive fractals.

These fractals have analytically known dimensions and size distributions,
making them ideal for deterministic testing of the objscale functions.

All objects are squares at various scales. The ensemble dimension and
size distribution exponent are determined by: D = log(n_zeros) / log(seed_size).
"""

import numpy as np
import sys
sys.path.insert(0, '../..')
import objscale


# Fractal seeds for recursive generation
# At each iteration: 0s are replaced with seed, 1s become solid blocks
# All objects are squares at various scales
FRACTAL_SEEDS = {
    # 3x3 center: inverted Sierpinski, D = log(8)/log(3) ≈ 1.893
    '3x3_center': np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.int32),
    # 5x5 center: D = log(24)/log(5) ≈ 1.975
    '5x5_center': np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=np.int32),
    # 5x5 block: D = log(16)/log(5) ≈ 1.723
    '5x5_block':  np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=np.int32),
}


def generate_fractal(seed='3x3_center', iterations=5):
    """
    Generate fractal using seed-based recursive construction.

    At each iteration, every 0 in the pattern is replaced with the seed,
    and every 1 is replaced with a solid block of 1s. This produces
    square objects at various scales with power-law size distribution.

    Parameters
    ----------
    seed : str or ndarray
        Seed pattern name ('3x3_center', '5x5_center', '5x5_block') or
        a 2D array. Default '3x3_center' produces inverted Sierpinski carpet.
    iterations : int
        Number of recursive iterations.
        - '3x3_center': size = 3^iterations (5 iter -> 243x243)
        - '5x5_*': size = 5^iterations (4 iter -> 625x625)

    Returns
    -------
    ndarray
        Binary array where 1s are the objects (squares at various scales).
    """
    if isinstance(seed, str):
        seed_arr = FRACTAL_SEEDS[seed].copy()
    else:
        seed_arr = np.asarray(seed, dtype=np.int32)

    seed_size = seed_arr.shape[0]
    pattern = seed_arr.copy()

    for _ in range(iterations - 1):
        current_size = pattern.shape[0]
        new_size = current_size * seed_size
        new_pattern = np.zeros((new_size, new_size), dtype=np.int32)

        for i in range(current_size):
            for j in range(current_size):
                block_i = i * seed_size
                block_j = j * seed_size
                if pattern[i, j] == 0:
                    # Replace with seed
                    new_pattern[block_i:block_i+seed_size, block_j:block_j+seed_size] = seed_arr
                else:
                    # Replace with solid block
                    new_pattern[block_i:block_i+seed_size, block_j:block_j+seed_size] = 1

        pattern = new_pattern

    return pattern


def expected_ensemble_dimension(seed='3x3_center'):
    """
    Calculate expected ensemble dimension from seed.

    D = log(n_zeros) / log(seed_size)

    Parameters
    ----------
    seed : str or ndarray
        Seed pattern name or array.

    Returns
    -------
    float
        Expected fractal dimension.
    """
    if isinstance(seed, str):
        seed_arr = FRACTAL_SEEDS[seed]
    else:
        seed_arr = np.asarray(seed)
    s = seed_arr.shape[0]
    n_zeros = np.sum(seed_arr == 0)
    return np.log(n_zeros) / np.log(s)


def expected_size_distribution_exponent(seed='3x3_center'):
    """
    Calculate expected size distribution exponent from seed.

    Returns the exponent for LENGTH dimensions: beta_L = log(n_zeros) / log(seed_size).
    For area, multiply by 0.5: beta_A = 0.5 * beta_L.

    Parameters
    ----------
    seed : str or ndarray
        Seed pattern name or array.

    Returns
    -------
    float
        Expected size distribution exponent for length dimensions (positive).
    """
    if isinstance(seed, str):
        seed_arr = FRACTAL_SEEDS[seed]
    else:
        seed_arr = np.asarray(seed)
    s = seed_arr.shape[0]
    n_zeros = np.sum(seed_arr == 0)
    return np.log(n_zeros) / np.log(s)


def default_iterations(seed):
    """Return default iterations: 7 for 3x3, 5 for 5x5."""
    if isinstance(seed, str):
        return 7 if seed.startswith('3x3') else 5
    else:
        seed_arr = np.asarray(seed)
        return 7 if seed_arr.shape[0] == 3 else 5


# =============================================================================
# Test Functions
# =============================================================================

def test_ensemble_box_dimension(seed='3x3_center', iterations=None, tolerance=0.05):
    """Test ensemble box dimension."""
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)

    dim, error = objscale.ensemble_box_dimension(
        [fractal], min_box_size=10, min_pixels=30
    )

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  box_dimension ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Box dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


def test_ensemble_correlation_dimension(seed='3x3_center', iterations=None, tolerance=0.05):
    """Test ensemble correlation dimension."""
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)

    dim, error = objscale.ensemble_correlation_dimension(
        [fractal], point_reduction_factor=10000,
        minlength=10, maxlength=100, interior_circles_only=False
    )

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  correlation_dimension ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Correlation dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


def test_size_distribution(seed='3x3_center', iterations=None, tolerance=0.001):
    """
    Test size distribution exponents for area, height, width, perimeter.

    The expected exponent for length dimensions is log(n_zeros)/log(seed_size).
    For area, multiply by 0.5.
    """
    if iterations is None:
        iterations = default_iterations(seed)
    beta_length = expected_size_distribution_exponent(seed)
    beta_area = 0.5 * beta_length

    print(f"\n=== Test: Size Distribution Exponents ({seed}) ===")

    fractal = generate_fractal(seed=seed, iterations=iterations)

    # Test all metrics
    metrics = {
        'area': beta_area,
        'perimeter': beta_length,
        'height': beta_length,
        'width': beta_length,
    }

    results = {}
    for metric, expected_beta in metrics.items():
        exponent, error = objscale.finite_array_powerlaw_exponent(
            [fractal], metric, bins=10000, min_threshold=10, min_count_threshold=1
        )

        diff = abs(exponent - expected_beta)
        passed = diff < tolerance
        results[metric] = passed

        status = "PASS" if passed else "FAIL"
        print(f"  {metric}: expected={expected_beta:.4f}, actual={exponent:.4f}, {status}")

    # Check all passed
    failed_metrics = [m for m, passed in results.items() if not passed]
    if failed_metrics:
        raise AssertionError(f"Size distribution failed for: {failed_metrics}")


def test_individual_fractal_dimension(seed='3x3_center', iterations=None, tolerance=0.01):
    """
    Test individual fractal dimension on square objects.

    Objects in the fractal are squares at various scales.
    For squares: P = 4*side and A = side^2, so P = 4*sqrt(A).
    The scaling P ~ sqrt(A)^D gives D = 1 (non-fractal 1D boundary).
    """
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = 1.0  # Squares have 1D (non-fractal) boundaries
    fractal = generate_fractal(seed=seed, iterations=iterations)

    dim, error = objscale.individual_fractal_dimension([fractal])

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  individual_fractal_dimension ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Individual dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


# =============================================================================
# Main test runner
# =============================================================================

def run_all_tests():
    """Run all automated tests."""
    print("=" * 60)
    print("OBJSCALE AUTOMATED FRACTAL DIMENSION TESTS")
    print("=" * 60)
    print("Using seed-based recursive fractals")

    passed = 0
    failed = 0

    for seed in ['3x3_center', '5x5_center', '5x5_block']:
        tests = [
            lambda s=seed: test_ensemble_box_dimension(seed=s),
            lambda s=seed: test_ensemble_correlation_dimension(seed=s),
            lambda s=seed: test_size_distribution(seed=s),
            lambda s=seed: test_individual_fractal_dimension(seed=s),
        ]

        for test_func in tests:
            try:
                test_func()
                passed += 1
            except AssertionError as e:
                print(f"FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR: {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
