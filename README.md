# DESI DR2 QSO redshift-error smearing

Portable NumPy/JAX implementation of the measured QSO redshift-error PDF
and residual Gaussian damping for the DESI DR2 full-shape mock challenge.

## Contents

- `PDF_smearing_QSO_z0.8-2.1_dv-kms.npz`: measured velocity-error PDF.
- `redshift_error_window.py`: reusable NumPy/JAX implementation.
- `QSO_redshift_error_smearing_example.ipynb`: model-independent example
  for power-spectrum and Sugiyama-bispectrum multipoles.

## Required operation order

1. Evaluate the full anisotropic P(k,mu) or B(k1,k2,k3).
2. Apply the measured-PDF and residual-Gaussian damping.
3. Project onto multipoles.
4. Apply the survey window.
5. Evaluate chi2.

Use observed/fiducial k_parallel in the smearing kernel. Keep aH_fid fixed
to the fiducial cosmology used to construct the catalogue coordinates.

Do not also use a survey window containing the same measured-PDF
correction.

## Current nuisance prior

vsmear ~ Uniform(0, 50) Mpc/h

## DESI environment

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
