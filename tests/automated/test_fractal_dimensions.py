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
        [fractal], min_box_size=10, max_box_size=64
    )

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  box_dimension ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Box dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


def test_ensemble_correlation_dimension(seed='3x3_center', iterations=None, tolerance=0.01):
    """Test ensemble correlation dimension."""
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)

    dim, error = objscale.ensemble_correlation_dimension(
        [fractal], point_reduction_factor=500,
        minlength=10, maxlength=300, interior_circles_only=True
    )

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  correlation_dimension ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Correlation dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


def test_ensemble_correlation_dimension_nonuniform(seed='3x3_center', iterations=None, tolerance=0.01):
    """Test ensemble correlation dimension with non-uniform pixel sizes.

    Tests several configurations where pixel resolution is increased along
    one axis while keeping the same physical dimensions:
    - Uniform scaling: dx=0.5, dy=0.5, dx=1/3, dy=1/3
    - Mid-array resolution change: dx halves or dy halves partway through
    The derived dimension should match the expected fractal dimension.
    """
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)
    nrows, ncols = fractal.shape

    # --- Uniform non-uniform pixel sizes ---
    uniform_configs = [
        ('dx=0.5', 0.5, 1.0),
        ('dy=0.5', 1.0, 0.5),
        ('dx=1/3', 1/3, 1.0),
        ('dy=1/3', 1.0, 1/3),
    ]

    for label, dx, dy in uniform_configs:
        if dx < 1.0:
            factor = int(round(1.0 / dx))
            stretched = np.repeat(fractal, factor, axis=1)
        elif dy < 1.0:
            factor = int(round(1.0 / dy))
            stretched = np.repeat(fractal, factor, axis=0)
        else:
            stretched = fractal

        x_sizes = np.full(stretched.shape, dx, dtype=np.float64)
        y_sizes = np.full(stretched.shape, dy, dtype=np.float64)

        dim, error = objscale.ensemble_correlation_dimension(
            [stretched], x_sizes=x_sizes, y_sizes=y_sizes,
            point_reduction_factor=500,
            minlength=10, maxlength=300, interior_circles_only=True
        )

        diff = abs(dim - expected_D)
        passed = diff < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  correlation_nonuniform ({seed}, {label}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

        assert passed, (
            f"Correlation dimension {dim:.4f} with {label} differs from "
            f"expected {expected_D:.4f} by more than {tolerance}"
        )

    # --- Mid-array resolution change ---
    # dx halves halfway through columns
    mid_col = ncols // 2
    left = fractal[:, :mid_col]
    right_stretched = np.repeat(fractal[:, mid_col:], 2, axis=1)
    stretched = np.concatenate([left, right_stretched], axis=1)
    x_sizes = np.ones(stretched.shape, dtype=np.float64)
    x_sizes[:, left.shape[1]:] = 0.5
    y_sizes = np.ones(stretched.shape, dtype=np.float64)

    dim, error = objscale.ensemble_correlation_dimension(
        [stretched], x_sizes=x_sizes, y_sizes=y_sizes,
        point_reduction_factor=500,
        minlength=10, maxlength=300, interior_circles_only=True
    )
    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  correlation_nonuniform ({seed}, dx_halves_mid): expected={expected_D:.4f}, actual={dim:.4f}, {status}")
    assert passed, (
        f"Correlation dimension {dim:.4f} with dx_halves_mid differs from "
        f"expected {expected_D:.4f} by more than {tolerance}"
    )

    # dy halves halfway through rows
    mid_row = nrows // 2
    top = fractal[:mid_row, :]
    bottom_stretched = np.repeat(fractal[mid_row:, :], 2, axis=0)
    stretched = np.concatenate([top, bottom_stretched], axis=0)
    x_sizes = np.ones(stretched.shape, dtype=np.float64)
    y_sizes = np.ones(stretched.shape, dtype=np.float64)
    y_sizes[top.shape[0]:, :] = 0.5

    dim, error = objscale.ensemble_correlation_dimension(
        [stretched], x_sizes=x_sizes, y_sizes=y_sizes,
        point_reduction_factor=500,
        minlength=10, maxlength=300, interior_circles_only=True
    )
    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  correlation_nonuniform ({seed}, dy_halves_mid): expected={expected_D:.4f}, actual={dim:.4f}, {status}")
    assert passed, (
        f"Correlation dimension {dim:.4f} with dy_halves_mid differs from "
        f"expected {expected_D:.4f} by more than {tolerance}"
    )


def test_ensemble_correlation_dimension_nonuniform_integer_repeats(
    seed='3x3_center', iterations=None, tolerance=0.01
):
    """Test correlation dimension with irregular integer repeat factors.

    Builds an equivalent fractal in physical space by repeating rows/columns by
    integer factors and assigning reciprocal pixel sizes to the repeated pixels.
    """
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)
    nrows, ncols = fractal.shape

    x_repeat_pattern = np.array([1, 2, 1, 3], dtype=np.int32)
    y_repeat_pattern = np.array([2, 1, 3, 1], dtype=np.int32)
    x_repeats = x_repeat_pattern[np.arange(ncols) % x_repeat_pattern.size]
    y_repeats = y_repeat_pattern[np.arange(nrows) % y_repeat_pattern.size]

    stretched = np.repeat(np.repeat(fractal, y_repeats, axis=0), x_repeats, axis=1)

    x_sizes_1d = np.concatenate([np.full(f, 1.0 / f) for f in x_repeats]).astype(np.float64)
    y_sizes_1d = np.concatenate([np.full(f, 1.0 / f) for f in y_repeats]).astype(np.float64)
    x_sizes = np.tile(x_sizes_1d, (stretched.shape[0], 1))
    y_sizes = np.tile(y_sizes_1d[:, None], (1, stretched.shape[1]))

    # Keep deterministic behavior despite random center subsampling.
    random_state = np.random.get_state()
    np.random.seed(0)
    try:
        dim, error = objscale.ensemble_correlation_dimension(
            [stretched], x_sizes=x_sizes, y_sizes=y_sizes,
            point_reduction_factor=500,
            minlength=10, maxlength=300, interior_circles_only=True
        )
    finally:
        np.random.set_state(random_state)

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  correlation_integer_repeats ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, (
        f"Correlation dimension {dim:.4f} with integer repeat-resolved pixel sizes "
        f"differs from expected {expected_D:.4f} by more than {tolerance}"
    )


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

    # Test unbinned
    dim, error = objscale.individual_fractal_dimension([fractal], bins=None)

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  individual_fractal_dimension unbinned ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Individual dimension (unbinned) {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"

    # Test binned (default bins=30, but use 1000 for discrete-size fractals)
    dim_b, error_b = objscale.individual_fractal_dimension([fractal], bins=1000)

    diff_b = abs(dim_b - expected_D)
    passed_b = diff_b < tolerance
    status_b = "PASS" if passed_b else "FAIL"
    print(f"  individual_fractal_dimension binned ({seed}): expected={expected_D:.4f}, actual={dim_b:.4f}, {status_b}")

    assert passed_b, f"Individual dimension (binned) {dim_b:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


def test_individual_fractal_dimension_nan_surrounded(tolerance=0.01):
    """
    Test individual fractal dimension with NaN-surrounded structures.

    Constructs a small array containing:
    - A normal 5x5 square cloud surrounded by 0s
    - A normal 7x7 square cloud surrounded by 0s
    - A single-pixel cloud whose 4-connected neighbors are all NaN

    The NaN-surrounded pixel has well-defined area but zero perimeter
    (perimeter edges adjacent to NaN are excluded). This must not cause
    an IndexError from mismatched area/perimeter array lengths.

    The two normal squares have P = 4*side, A = side^2, so D = 1.
    """
    arr = np.full((40, 40), np.nan)
    arr[2:-2, 2:-2] = 0

    # Normal squares at various sizes (need >= 3 for regression)
    arr[4:7, 4:7] = 1       # 3x3: A=9, P=12
    arr[4:9, 12:17] = 1     # 5x5: A=25, P=20
    arr[4:11, 22:29] = 1    # 7x7: A=49, P=28
    arr[15:25, 4:14] = 1    # 10x10: A=100, P=40

    # Single pixel surrounded by NaN (cross of NaN neighbors)
    arr[30, 20] = 1
    arr[29, 20] = np.nan
    arr[31, 20] = np.nan
    arr[30, 19] = np.nan
    arr[30, 21] = np.nan

    expected_D = 1.0

    # Unbinned
    dim, error = objscale.individual_fractal_dimension([arr], min_a=1, bins=None)

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  individual_fractal_dimension_nan_surrounded (unbinned): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, (
        f"Individual dimension (nan-surrounded, unbinned) {dim:.4f} differs "
        f"from expected {expected_D:.4f} by more than {tolerance}"
    )

    # Binned
    dim_b, error_b = objscale.individual_fractal_dimension([arr], min_a=1, bins=1000)

    diff_b = abs(dim_b - expected_D)
    passed_b = diff_b < tolerance
    status_b = "PASS" if passed_b else "FAIL"
    print(f"  individual_fractal_dimension_nan_surrounded (binned): expected={expected_D:.4f}, actual={dim_b:.4f}, {status_b}")

    assert passed_b, (
        f"Individual dimension (nan-surrounded, binned) {dim_b:.4f} differs "
        f"from expected {expected_D:.4f} by more than {tolerance}"
    )


def test_ensemble_box_renyi_dimension_q0(seed='3x3_center', iterations=None, tolerance=0.05):
    """Renyi q=0 should match the analytical box dimension and the
    standalone ensemble_box_dimension wrapper exactly."""
    if iterations is None:
        iterations = default_iterations(seed)
    expected_D = expected_ensemble_dimension(seed)
    fractal = generate_fractal(seed=seed, iterations=iterations)

    dim, error = objscale.ensemble_box_renyi_dimension(
        [fractal], q=0.0, min_box_size=10, max_box_size=64
    )

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  renyi_dimension q=0 ({seed}): expected={expected_D:.4f}, actual={dim:.4f}, {status}")
    assert passed, f"Renyi q=0 dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"

    # Consistency: must equal the standalone ensemble_box_dimension wrapper exactly.
    box_d, _ = objscale.ensemble_box_dimension(
        [fractal], min_box_size=10, max_box_size=64
    )
    assert abs(dim - box_d) < 1e-10, (
        f"renyi(q=0) {dim:.10f} != ensemble_box_dimension {box_d:.10f}"
    )
    print(f"  renyi_dimension q=0 ({seed}): matches ensemble_box_dimension wrapper, PASS")


def test_ensemble_box_renyi_dimension_all_ones_analytic():
    """Analytic oracle: an all-ones array has D_q = 2 exactly for any q.

    For an all-ones array of side n with box size F, every box has n_i = F^2,
    so all p_i = F^2/n^2 are equal and Z_q^(p) = N_b * (F^2/n^2)^q = (n^2/F^2)^(1-q).
    Then log Z_q vs log F has slope 2*(q-1), giving D_q = 2 exactly. The
    Shannon-entropy form at q=1 likewise gives -log(N_b) ~ -2 log F + const,
    slope -2 = -D_1 → D_1 = 2.

    Also probes the mixed-size-ensemble code path that the codex review
    flagged as silently dropping smaller arrays at large box sizes (the bug
    biased D_q upward to ~2.026 on a 64+16 ensemble).
    """
    # Single array
    arr = np.ones((128, 128), dtype=np.float64)
    for q in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        D, _ = objscale.ensemble_box_renyi_dimension(
            [arr], q=q, set='ones',
            box_sizes=[2, 4, 8, 16, 32, 64], min_box_size=2,
        )
        assert abs(D - 2.0) < 1e-10, (
            f"all-ones single array: q={q} gave D={D:.10f}, expected exactly 2.0"
        )
    print("  renyi_all_ones single array: D_q == 2 exactly for q in {0,0.5,1,1.5,2,3}, PASS")

    # Mixed-size ensemble (codex's regression case)
    a = np.ones((64, 64), dtype=np.float64)
    b = np.ones((16, 16), dtype=np.float64)
    for q in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        D, _ = objscale.ensemble_box_renyi_dimension(
            [a, b], q=q, set='ones',
            box_sizes=[2, 4, 8, 16], min_box_size=2,
        )
        assert abs(D - 2.0) < 1e-10, (
            f"all-ones mixed-size ensemble: q={q} gave D={D:.10f}, expected exactly 2.0"
        )
    print("  renyi_all_ones mixed-size ensemble: D_q == 2 exactly for q in {0,0.5,1,1.5,2,3}, PASS")

    # Vector q on mixed-size ensemble
    qs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    D_vec, _ = objscale.ensemble_box_renyi_dimension(
        [a, b], q=qs, set='ones',
        box_sizes=[2, 4, 8, 16], min_box_size=2,
    )
    for q, d in zip(qs, D_vec):
        assert abs(d - 2.0) < 1e-10, (
            f"all-ones vector q: q={q} gave D={d:.10f}, expected exactly 2.0"
        )
    print("  renyi_all_ones vector q on mixed-size ensemble: PASS")


def _synthetic_fbm_level_set(size, H, seed):
    """Pure-numpy 2D fBm via spectral synthesis, thresholded at the median.

    For a fBm field with Hurst H, the level set has D_q = 2 - H for all q
    (it is a monofractal — D_q is independent of q). This is the cleanest
    deterministic test for the Rényi-dimension family.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(size).reshape(1, -1)
    ky = np.fft.fftfreq(size).reshape(-1, 1)
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0  # avoid divide-by-zero at DC
    # 2D fBm spectral exponent for amplitudes is -(H + 1).
    spectrum = k ** (-(H + 1.0))
    spectrum[0, 0] = 0.0  # zero mean
    noise = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    field = np.fft.ifft2(spectrum * noise).real
    return (field > np.median(field)).astype(np.float64)


def test_ensemble_box_renyi_dimension_fbm_monofractal(tolerance=0.15):
    """For a 2D fBm level set the dimension is D_q = 2 - H for all q.

    Smoke test for the q-vector path on a true monofractal. The tolerance
    is loose because box-counting on a 512^2 fBm has well-known finite-size
    bias of order 0.05-0.1; tighter accuracy needs much larger fields. The
    real validation lives in the standalone experiment script. Here we
    care that:
      * the q-vector path runs end-to-end and returns finite numbers
      * the scalar wrappers (ensemble_box_dimension, ensemble_information_dimension)
        agree exactly with the corresponding vector entries
      * the answer is in the right ballpark
    """
    H = 0.3
    expected_D = 2.0 - H
    arr = _synthetic_fbm_level_set(size=512, H=H, seed=0)

    qs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    D_arr, err_arr = objscale.ensemble_box_renyi_dimension(
        [arr], q=qs, min_box_size=4, max_box_size=64
    )

    for q, d in zip(qs, D_arr):
        diff = abs(d - expected_D)
        passed = diff < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  renyi_fbm q={q}: expected~{expected_D:.4f}, actual={d:.4f}, {status}")
        assert passed, (
            f"Renyi q={q} on fBm gave {d:.4f}, expected ~{expected_D:.4f} +/- {tolerance}"
        )

    # Consistency: q=0 entry must equal the standalone ensemble_box_dimension wrapper.
    box_d, _ = objscale.ensemble_box_dimension(
        [arr], min_box_size=4, max_box_size=64
    )
    assert abs(D_arr[0] - box_d) < 1e-10, (
        f"renyi(q=0) {D_arr[0]:.10f} != ensemble_box_dimension {box_d:.10f}"
    )

    # Consistency: q=1 entry must equal ensemble_information_dimension(method='box').
    # (The default method is now 'sandbox', which uses a different estimator;
    # we check the box wrapper here because that's what we're consistency-checking.)
    info_d, _ = objscale.ensemble_information_dimension(
        [arr], method='box', min_box_size=4, max_box_size=64
    )
    assert abs(D_arr[2] - info_d) < 1e-10, (
        f"renyi(q=1) {D_arr[2]:.10f} != ensemble_information_dimension(method='box') {info_d:.10f}"
    )
    print(f"  renyi_fbm consistency: scalar wrappers match vector entries, PASS")


def test_ensemble_sandbox_renyi_dimension_analytic_oracle():
    """Analytic oracle: an all-ones array has D_q = 2 for any q.

    For an all-ones array of side n, the set is the full plane (filled
    square), which has dimension exactly 2. The sandbox estimator should
    recover D_q = 2 for any q to within finite-size tolerance.

    Tighter than the recursive-fractal tolerance because there's no
    boundary irregularity to bias the slope.
    """
    arr = np.ones((128, 128), dtype=np.float64)
    qs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    D, err = objscale.ensemble_sandbox_renyi_dimension(
        [arr], q=qs, set='ones',
        minlength=2, maxlength=40, nbins=20,
    )
    for q, d in zip(qs, D):
        diff = abs(d - 2.0)
        passed = diff < 0.05
        status = "PASS" if passed else "FAIL"
        print(f"  sandbox_oracle q={q}: D={d:.4f}, expected~2.000, {status}")
        assert passed, (
            f"sandbox D_{q} on all-ones {d:.4f} differs from 2.0 by more than 0.05"
        )

    # Scalar q convention
    D_sc, _ = objscale.ensemble_sandbox_renyi_dimension(
        arr, q=2.0, set='ones',
        minlength=2, maxlength=40, nbins=20,
    )
    assert abs(D_sc - D[3]) < 1e-10, (
        f"scalar q=2 ({D_sc:.10f}) != vector q=2 entry ({D[3]:.10f})"
    )
    print("  sandbox_oracle scalar/vector consistency: PASS")


def test_ensemble_sandbox_renyi_dimension_fbm_monofractal(tolerance=0.15):
    """For a 2D fBm level set the dimension is D_q = 2 - H for all q.

    Loose tolerance because we use a tiny 256^2 field for test speed; the
    real validation is in the standalone experiment script.
    """
    H = 0.3
    expected_D = 2.0 - H
    arr = _synthetic_fbm_level_set(size=256, H=H, seed=0)

    qs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    D_arr, err_arr = objscale.ensemble_sandbox_renyi_dimension(
        [arr], q=qs, set='edge',
        minlength=2, maxlength=80, nbins=25,
    )

    for q, d in zip(qs, D_arr):
        diff = abs(d - expected_D)
        passed = diff < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  sandbox_fbm q={q}: expected~{expected_D:.4f}, actual={d:.4f}, {status}")
        assert passed, (
            f"sandbox q={q} on fBm gave {d:.4f}, expected ~{expected_D:.4f} +/- {tolerance}"
        )


def test_sandbox_gp_equivalence_at_q2():
    """At q=2, sandbox returns the same dimension as ensemble_correlation_dimension.

    Since ensemble_correlation_dimension is now a thin wrapper around
    ensemble_sandbox_renyi_dimension(q=2), the two should agree to float
    precision when called with the same parameters and (crucially) no
    random subsampling.
    """
    arr = _synthetic_fbm_level_set(size=256, H=0.3, seed=0)

    D_sandbox, err_sandbox, bins_sandbox, Z_sandbox = objscale.ensemble_sandbox_renyi_dimension(
        [arr], q=2.0, set='edge',
        minlength=2, maxlength=80, nbins=25,
        point_reduction_factor=1, return_values=True,
    )
    D_gp, err_gp, bins_gp, Cl_gp = objscale.ensemble_correlation_dimension(
        [arr],
        minlength=2, maxlength=80, nbins=25,
        point_reduction_factor=1, return_C_l=True,
    )

    assert abs(D_sandbox - D_gp) < 1e-12, (
        f"sandbox D_2 {D_sandbox:.12f} != GP D_2 {D_gp:.12f}"
    )
    assert np.allclose(bins_sandbox, bins_gp), "bins differ between sandbox and GP"
    assert np.allclose(Z_sandbox, Cl_gp), "Z[q=2] != C_l (the partition function should match)"
    print(f"  sandbox_gp_equivalence: D_2={D_sandbox:.6f} matches GP exactly, PASS")


def test_sandbox_duplicate_q1():
    """Multiple q==1 entries in the q vector must all be handled.

    Regression test for a codex-flagged bug where only the first q==1
    entry got the log-form special case; later duplicates fell through to
    the M^0 path and divided by zero (returning inf or garbage). The fix
    replaces the single q1_index with a per-element mask.
    """
    arr = _synthetic_fbm_level_set(size=256, H=0.3, seed=0)
    D, err = objscale.ensemble_sandbox_renyi_dimension(
        [arr], q=np.array([0.5, 1.0, 1.0, 2.0]), set='edge',
        minlength=2, maxlength=80, nbins=25,
    )
    assert abs(D[1] - D[2]) < 1e-12, (
        f"duplicate q=1 entries disagree: {D[1]:.12f} vs {D[2]:.12f}"
    )
    assert np.isfinite(D[1]) and abs(D[1] - 1.7) < 0.15, (
        f"q=1 entry {D[1]:.4f} not near theory 1.7"
    )
    print(f"  sandbox duplicate q=1: both entries = {D[1]:.4f}, PASS")


def test_information_dimension_method_parameter():
    """ensemble_information_dimension dispatches to sandbox by default.

    Verifies (a) the default method is 'sandbox', (b) both methods return
    finite D_1 values close to the theoretical fBm value, (c) explicit
    method='sandbox' matches the default.
    """
    arr = _synthetic_fbm_level_set(size=256, H=0.3, seed=0)
    expected_D = 1.7

    D_default, _ = objscale.ensemble_information_dimension(
        [arr], minlength=2, maxlength=80, nbins=25,
    )
    D_sandbox, _ = objscale.ensemble_information_dimension(
        [arr], method='sandbox', minlength=2, maxlength=80, nbins=25,
    )
    D_box, _ = objscale.ensemble_information_dimension(
        [arr], method='box', min_box_size=4, max_box_size=64,
    )

    assert abs(D_default - D_sandbox) < 1e-12, (
        f"default ({D_default:.6f}) != explicit sandbox ({D_sandbox:.6f}); "
        "default method should be 'sandbox'"
    )
    assert abs(D_sandbox - expected_D) < 0.15, (
        f"sandbox D_1 {D_sandbox:.4f} differs from theory {expected_D:.4f}"
    )
    assert abs(D_box - expected_D) < 0.15, (
        f"box D_1 {D_box:.4f} differs from theory {expected_D:.4f}"
    )
    print(
        f"  information_dim method dispatch: default==sandbox={D_sandbox:.4f}, "
        f"box={D_box:.4f}, both ~ {expected_D}, PASS"
    )

    # Bad method
    try:
        objscale.ensemble_information_dimension([arr], method='garbage')
        raise AssertionError("Expected ValueError for method='garbage'")
    except ValueError as e:
        assert 'sandbox' in str(e) and 'box' in str(e)
    print("  information_dim bad method: correctly raised ValueError, PASS")


def test_individual_correlation_dimension(tolerance=0.05):
    """
    Test individual correlation dimension on a 1D line (wide "eye" shape).

    A single row of pixels has boundary points forming a 1D set,
    so the correlation dimension should be exactly 1.
    """
    # Build a wide eye: NaN border, zeros inside, one row of 1s
    width = 2000
    arr = np.full((5, width), np.nan)
    arr[1:-1, 1:-1] = 0
    arr[2, 2:-2] = 1  # single row of pixels

    expected_D = 1.0
    dim, error = objscale.individual_correlation_dimension(arr, n=1)

    diff = abs(dim - expected_D)
    passed = diff < tolerance
    status = "PASS" if passed else "FAIL"
    print(f"  individual_correlation_dimension (eye): expected={expected_D:.4f}, actual={dim:.4f}, {status}")

    assert passed, f"Individual correlation dimension {dim:.4f} differs from expected {expected_D:.4f} by more than {tolerance}"


# =============================================================================
# Main test runner
# =============================================================================

def test_correlation_dimension_maxlength_too_large():
    """Test that maxlength too large for interior_circles_only raises ValueError."""
    arr = (np.random.random((64, 64)) < 0.3).astype(float)
    try:
        objscale.ensemble_correlation_dimension(
            arr, maxlength=32, interior_circles_only=True
        )
        raise AssertionError(
            "Expected ValueError for maxlength=32 on 64x64 array with "
            "interior_circles_only=True, but no error was raised"
        )
    except ValueError as e:
        assert 'interior_circles_only' in str(e)
        print(f"  correlation_dimension_maxlength_too_large: correctly raised ValueError, PASS")


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
            lambda s=seed: test_ensemble_box_renyi_dimension_q0(seed=s),
            lambda s=seed: test_ensemble_correlation_dimension(seed=s),
            lambda s=seed: test_ensemble_correlation_dimension_nonuniform(seed=s),
            lambda s=seed: test_size_distribution(seed=s),
            lambda s=seed: test_individual_fractal_dimension(seed=s),
        ]

        if seed == '3x3_center':
            tests.append(lambda s=seed: test_ensemble_correlation_dimension_nonuniform_integer_repeats(seed=s))

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

    # Seed-independent tests
    standalone_tests = [
        test_individual_fractal_dimension_nan_surrounded,
        test_individual_correlation_dimension,
        test_correlation_dimension_maxlength_too_large,
        test_ensemble_box_renyi_dimension_all_ones_analytic,
        test_ensemble_box_renyi_dimension_fbm_monofractal,
        test_ensemble_sandbox_renyi_dimension_analytic_oracle,
        test_ensemble_sandbox_renyi_dimension_fbm_monofractal,
        test_sandbox_gp_equivalence_at_q2,
        test_sandbox_duplicate_q1,
        test_information_dimension_method_parameter,
    ]
    for test_func in standalone_tests:
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
