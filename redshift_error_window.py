"""Portable redshift-error windows for a full pre-projection P/B prediction.

The PDF is a distribution of line-of-sight velocity errors ``dv`` in km/s.
Its Fourier transform is tabulated once. During a fit, only ``vsmear`` varies.
Both ``k_parallel`` and ``vsmear`` use fiducial h/Mpc and Mpc/h units, so
their product is dimensionless. Use the *observed/fiducial* k, before AP
rescaling, angular projection, and survey-window convolution.

Example (same fiducial convention as desi-clustering)::

    from cosmoprimo.fiducial import DESI
    from redshift_error_window import make_windows

    z_eff = 1.48  # Replace with the effective redshift of your window.
    aH_fid = 100.0 * DESI().efunc(z_eff) / (1.0 + z_eff)
    pk_window, bk_window = make_windows("PDF_smearing_QSO_z0.8-2.1_dv-kms.npz", aH_fid)

    # pk_full and bk_full include every term, stochastic contributions too.
    pk_smeared = pk_window(pk_full, k[:, None] * mu[None, :], vsmear)
    bk_smeared = bk_window(bk_full, k1mu1, k2mu2, k3mu3, vsmear)

The returned functions work with ``jax.jit`` when backend="jax". Construct
them once, outside the likelihood evaluation loop; keep vsmear a dynamic
argument. A group without JAX can choose backend="numpy" instead.
"""

import argparse
from pathlib import Path

import numpy as np


def tabulate_pdf_damping(pdf_file, aH_fid, qmax=1.0, nq=2049):
    """Return grids ``(q, D_pdf(q))`` for a velocity-error PDF.

    ``aH_fid = 100 E_fid(z_eff) / (1 + z_eff)`` converts km/s to Mpc/h.
    The PDF file must contain one-dimensional ``dv`` and ``pdf`` arrays.
    NPZ files must also declare ``dv_unit='km/s'``; HDF5 files follow the
    current DESI convention that the ``dv`` dataset is in km/s.
    As in the current DESI implementation, we use the real cosine transform:
    this assumes an effectively symmetric, zero-centered error distribution.
    Choose qmax above every |k_parallel| in the P and B angular grids.
    """
    if not np.isfinite(aH_fid) or aH_fid <= 0:
        raise ValueError("aH_fid must be finite and positive")
    if not np.isfinite(qmax) or qmax <= 0 or nq < 2:
        raise ValueError("qmax must be positive and nq >= 2")

    pdf_file = Path(pdf_file)
    if pdf_file.suffix == ".npz":
        with np.load(pdf_file, allow_pickle=False) as file:
            if str(file["dv_unit"].item()) != "km/s":
                raise ValueError("NPZ dv_unit must be 'km/s'")
            dv = np.asarray(file["dv"], dtype=np.float64)
            pdf = np.asarray(file["pdf"], dtype=np.float64)
    elif pdf_file.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(pdf_file, "r") as file:
            dv = np.asarray(file["dv"], dtype=np.float64)
            pdf = np.asarray(file["pdf"], dtype=np.float64)
    else:
        raise ValueError("PDF file must be .npz, .h5, or .hdf5")
    if dv.ndim != 1 or pdf.shape != dv.shape or len(dv) < 2:
        raise ValueError("dv and pdf must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(dv)) or not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
        raise ValueError("dv/pdf contain non-finite values or a negative density")
    order = np.argsort(dv)
    dv, pdf = dv[order], pdf[order]
    widths = np.diff(dv)
    if np.any(widths <= 0):
        raise ValueError("dv coordinates must be distinct")

    # Trapezoidal integration also handles a nonuniform velocity grid.
    weights = np.empty_like(dv)
    weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2
    weights *= pdf
    norm = weights.sum()
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("PDF normalization must be positive")
    weights /= norm

    q = np.linspace(0.0, qmax, nq)
    damping = np.empty_like(q)
    # The supplied QSO PDF has many dv points; batching limits temporary RAM.
    for start in range(0, nq, 128):
        stop = min(start + 128, nq)
        phase = np.outer(q[start:stop] / aH_fid, dv)
        damping[start:stop] = np.cos(phase) @ weights
    damping[0] = 1.0
    return q, damping


def make_windows(pdf_file, aH_fid, *, qmax=1.0, nq=2049, backend="jax"):
    """Return ``(pk_window, bk_window)`` for a fixed PDF and free vsmear.

    These are *pre-projection operators*, not ready-made survey matrices.
    ``pk_window(pk_full, q, vsmear)`` expects q = k*mu and multiplies the
    complete anisotropic power spectrum by D(q)^2.
    ``bk_window(bk_full, q1, q2, q3, vsmear)`` multiplies the complete
    orientation-dependent bispectrum by D(q1)*D(q2)*D(q3), where the q values
    are the three line-of-sight wavevectors of a closed triangle.

    The user still performs angular projection and survey-window convolution
    afterwards. For q outside [-qmax, qmax] the result is NaN, so an
    undersized table cannot silently bias a fit.
    """
    q_grid, pdf_grid = tabulate_pdf_damping(pdf_file, aH_fid, qmax, nq)
    if backend == "jax":
        import jax.numpy as xp
    elif backend == "numpy":
        xp = np
    else:
        raise ValueError("backend must be 'jax' or 'numpy'")
    q_grid, pdf_grid = xp.asarray(q_grid), xp.asarray(pdf_grid)

    def single_field(q, vsmear):
        q = xp.asarray(q)
        pdf_part = xp.interp(xp.abs(q), q_grid, pdf_grid, right=xp.nan)
        gaussian_part = xp.exp(-0.5 * (q * vsmear) ** 2)
        return pdf_part * gaussian_part

    def pk_window(pk_full, q, vsmear):
        return pk_full * single_field(q, vsmear) ** 2

    def bk_window(bk_full, q1, q2, q3, vsmear):
        return (bk_full * single_field(q1, vsmear)
                * single_field(q2, vsmear) * single_field(q3, vsmear))

    return pk_window, bk_window


def main():
    """Print a tiny numerical example; fitting code should import make_windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_file", type=Path, help="NPZ or HDF5 file containing dv and pdf")
    parser.add_argument("--aH-fid", type=float, required=True,
                        help="100 * E_fid(z_eff) / (1 + z_eff), in km/s per Mpc/h")
    parser.add_argument("--vsmear", type=float, default=2.0, help="Gaussian scale in Mpc/h")
    args = parser.parse_args()

    pk_window, bk_window = make_windows(args.pdf_file, args.aH_fid, backend="numpy")
    q = np.array([0.0, 0.1, 0.2])  # Example line-of-sight k, in h/Mpc.
    unit_theory = np.ones_like(q)
    print("q [h/Mpc]   P damping   B damping for (q, -q, 0)")
    p = pk_window(unit_theory, q, args.vsmear)
    b = bk_window(unit_theory, q, -q, np.zeros_like(q), args.vsmear)
    for qi, pi, bi in zip(q, p, b):
        print(f"{qi:8.3f}   {pi:9.6f}   {bi:9.6f}")


if __name__ == "__main__":
    main()
