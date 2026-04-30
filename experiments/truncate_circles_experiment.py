"""Compare three boundary-handling modes for the sandbox correlation
integral on level sets of fBm.

Modes (corresponding to ``interior_circles_only`` values on the
``truncate-circles-experiment`` branch):

* ``True``       : centers must be at least ``maxlength`` from every edge.
                   Small circles near the boundary are excluded.
* ``False``      : every set point is a center, but circles that extend
                   past an edge are silently truncated (M_i biased low).
* ``'truncate'`` : every set point is a center, but each center contributes
                   only at radii ``r <= d_edge``. Per-bin partition function
                   is normalized by the number of admissible centers.

Generates 10 fBm 4096^2 with H=0.5, thresholds at 0 (theoretical level-set
edge dimension D_2 = 2 - H = 1.5), runs all three modes at q=2, and saves
the partition functions and a local-slope plot.
"""

import os
import time

import numpy as np
import scaleinvariance as si

from objscale import ensemble_sandbox_renyi_dimension


N_REALIZATIONS = int(os.environ.get('TC_N', 10))
SIZE = int(os.environ.get('TC_SIZE', 4096))
H = float(os.environ.get('TC_H', 0.5))
PRF = float(os.environ.get('TC_PRF', 1))
FIELD = os.environ.get('TC_FIELD', 'fbm').lower()  # 'fbm' or 'fif'
ALPHA = float(os.environ.get('TC_ALPHA', 2.0))
C1 = float(os.environ.get('TC_C1', 0.1))
NBINS = 50

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
if FIELD == 'fif':
    _TAG = f'fif_{N_REALIZATIONS}x{SIZE}_alpha{ALPHA:g}_C1{C1:g}_H{H:g}'
else:
    _TAG = f'{N_REALIZATIONS}x{SIZE}_H{H:g}'
if PRF != 1:
    _TAG += f'_prf{PRF:g}'
NPZ_PATH = os.path.join(OUT_DIR, f'truncate_circles_partition_{_TAG}.npz')
FIG_PATH = os.path.join(OUT_DIR, f'truncate_circles_local_slopes_{_TAG}.png')


def generate_fields():
    fields = []
    rng = np.random.default_rng(seed=0)
    for i in range(N_REALIZATIONS):
        # Each call uses fresh global noise; seed numpy for determinism.
        np.random.seed(rng.integers(2**31 - 1))
        if FIELD == 'fif':
            f = si.FIF_ND((SIZE, SIZE), alpha=ALPHA, C1=C1, H=H,
                          periodic=True)
            # FIF is a positive flux with mean 1; threshold at the per-
            # realization median so the binary set fraction is ~0.5,
            # matching fBm-at-0.
            thresh = float(np.median(f))
            binary = (f > thresh).astype(np.int8)
        else:
            f = si.fBm_ND_circulant((SIZE, SIZE), H=H)
            binary = (f > 0).astype(np.int8)
        del f
        fields.append(binary)
        print(f'  generated field {i + 1}/{N_REALIZATIONS} '
              f'(set fraction = {binary.mean():.3f})', flush=True)
    return fields


def run_mode(fields, mode):
    t0 = time.time()
    D, err, bins, Z = ensemble_sandbox_renyi_dimension(
        fields,
        q=2.0,
        set='edge',
        interior_circles_only=mode,
        nbins=NBINS,
        point_reduction_factor=PRF,
        return_values=True,
    )
    dt = time.time() - t0
    print(f'  mode={mode!r:>10s}  D2 = {D:.4f} +/- {err:.4f}  '
          f'(elapsed {dt:.1f}s)', flush=True)
    return bins, Z


def main():
    if os.path.exists(NPZ_PATH):
        print(f'loading cached partition functions from {NPZ_PATH}')
        data = np.load(NPZ_PATH)
        bins = data['bins']
        Z_interior = data['Z_interior']
        Z_open = data['Z_open']
        Z_truncate = data['Z_truncate']
    else:
        print('generating fields...')
        fields = generate_fields()

        print('computing sandbox partition functions, q=2:')
        bins, Z_interior = run_mode(fields, True)
        bins_o, Z_open = run_mode(fields, False)
        bins_t, Z_truncate = run_mode(fields, 'truncate')
        assert np.allclose(bins, bins_o) and np.allclose(bins, bins_t)

        np.savez(
            NPZ_PATH,
            bins=bins,
            Z_interior=Z_interior,
            Z_open=Z_open,
            Z_truncate=Z_truncate,
        )
        print(f'saved partition functions to {NPZ_PATH}')

    # ---- local-slope plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_x = np.log10(bins.astype(float))
    x_mid = 10.0 ** (0.5 * (log_x[:-1] + log_x[1:]))

    def local_D2(Z):
        # q=2, so slope of log10(Z) vs log10(r) IS D_2 directly.
        log_y = np.log10(np.where(Z > 0, Z, np.nan))
        return np.diff(log_y) / np.diff(log_x)

    D_int = local_D2(Z_interior)
    D_open = local_D2(Z_open)
    D_trunc = local_D2(Z_truncate)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(x_mid, D_int, '-', color='#3674B3', linewidth=1.8,
            label="interior_circles_only=True (centers restricted)")
    ax.plot(x_mid, D_open, '-', color='#B33683', linewidth=1.8,
            label="interior_circles_only=False (unrestricted, truncated)")
    ax.plot(x_mid, D_trunc, '-', color='#184727', linewidth=1.8,
            label="interior_circles_only='truncate' (per-center cap)")

    if FIELD == 'fif':
        ax.axhline(2.0 - H, linestyle='--', color='0.3', linewidth=1, alpha=0.7,
                   label=f'2 - H = {2.0 - H:.1f} (fBm reference)')
    else:
        ax.axhline(2.0 - H, linestyle='--', color='0.3', linewidth=1, alpha=0.7,
                   label=f'D = 2 - H = {2.0 - H:.1f}')
    ax.set_xscale('log')
    ax.set_ylim(1.30, 1.70)
    ax.set_xlabel(r'circle radius $l$ (pixels)', fontsize=11)
    ax.set_ylabel(r'local $D_2 = d\log Z_2 / d\log l$', fontsize=11)
    if FIELD == 'fif':
        title = (
            f'FIF alpha={ALPHA:g}, C1={C1:g}, H={H:g}, '
            f'median-threshold edge set, '
            f'{N_REALIZATIONS}x{SIZE}^2 ensemble'
        )
    else:
        title = (
            f'fBm H={H}, threshold-zero edge set, '
            f'{N_REALIZATIONS}x{SIZE}^2 ensemble'
        )
    if PRF != 1:
        title += f'  (prf={PRF:g})'
    ax.set_title(title, fontsize=11)
    ax.tick_params(direction='in', labelsize=9)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {FIG_PATH}')


if __name__ == '__main__':
    main()
