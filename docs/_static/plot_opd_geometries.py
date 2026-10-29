"""Illustration of the three OPD geometries for a top-hat spectrum (100–300 GHz)."""

import sys
import os
from pathlib import Path

# Add the ftsdata package root to sys.path BEFORE any local imports
# The package root is the parent of the 'ftsdata' package directory
_this_file = Path(__file__).resolve() if "__file__" in globals() else Path.cwd() / "docs" / "_static" / "plot_opd_geometries.py"
_pkg_root = _this_file.parents[2]  # /home/abeelen/Python/ftsdata

# Insert at front so it takes precedence over namespace package detection
sys.path.insert(0, str(_pkg_root))

# Also ensure the actual package directory is importable
os.chdir(_pkg_root)

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS

from ftsdata import (
    SingleSidedOpdData,
    DoubleSidedOpdData,
    OneSidedOpdData,
)

from ftsdata.tests.models import rect_os, rect_ss, rect_ds


# ─── Analytical interferogram of a top-hat spectrum ──────────────────
# Top-hat between 100 and 300 GHz
nu = 150 * u.GHz
sigma = 100 * u.GHz

# OPD sampling
dopd_mm = 0.2  # mm
opd_max_mm = 50  # mm

# ─── 1) Double-sided: symmetric around ZPD (0) ──────────────────────
itg_ds = rect_ds(nu, sigma, cdelt_opd=0.1 * u.mm, n_itg=int(50 * 2 / 0.1))

# ─── 2) One-sided: ZPD at the start (OPD ≥ 0) ───────────────────────
itg_os = rect_os(nu, sigma, cdelt_opd=0.1 * u.mm, n_itg=int(50 / 0.1))

# ─── 3) Single-sided (asymmetric): ZPD at 25 % from the start ───────
itg_ss = rect_ss(nu, sigma, cdelt_opd=0.1 * u.mm, n_itg=int(50 * (1 + 0.25) / 0.1))

# ─── Verify class dispatch ──────────────────────────────────────────
assert isinstance(itg_ds, DoubleSidedOpdData), f"Got {type(itg_ds)}"
assert isinstance(itg_os, OneSidedOpdData), f"Got {type(itg_os)}"
assert isinstance(itg_ss, SingleSidedOpdData), f"Got {type(itg_ss)}"

spec = itg_ds.to_spectra()

# ─── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex="col", constrained_layout=True)

# Colour per geometry
col_ds = "#1f77b4"
col_os = "#ff7f0e"
col_ss = "#2ca02c"

# Frequency axis for the spectrum
freq = np.linspace(0, 600, 6000)

# ── Top-hat spectrum ──
spectrum = np.where((freq >= 100) & (freq <= 200), 1.0, 0.0)
axes[0, 0].plot(freq, spectrum, "k-", lw=1.5)
axes[0, 0].axvspan(100, 200, color="C0", alpha=0.15)
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Input spectrum (top-hat 100–300 GHz)")
axes[0, 0].set_xlim(0, 600)
axes[0, 0].grid(alpha=0.3)

# ── ILS kernel (for the double-sided OPD sampling) ──
# sinc(x) = sin(pi*x)/(pi*x), first zero at x=1 → Δν = c/(2*OPD_max)
# For OPD_max = 50 mm, Δν ≈ 3 GHz
ils = np.abs(np.sinc(freq * dopd_mm * 2 * opd_max_mm / 299.792458))  # approx
axes[1, 0].plot(freq, ils / np.max(ils), "k-", lw=1.2)
axes[1, 0].set_xlabel("Frequency (GHz)")
axes[1, 0].set_ylabel("Normalised ILS")
axes[1, 0].set_title(f"ILS for OPD_max = {opd_max_mm} mm, ΔOPD = {dopd_mm} mm")
axes[1, 0].set_xlim(0, 400)
axes[1, 0].grid(alpha=0.3)

# ── Interferograms ──
for ax, itg, label, color in [
    (axes[2, 1], itg_ds, "DoubleSidedOpdData", col_ds),
    (axes[1, 1], itg_os, "OneSidedOpdData", col_os),
    (axes[0, 1], itg_ss, "SingleSidedOpdData", col_ss),
]:
    opd = itg.opd_axis.to_value(u.mm)
    data = itg.data
    ax.plot(opd, data, color=color, lw=1.2)
    ax.axvline(0, color="k", ls=":", lw=0.8, label="ZPD (OPD = 0)")
    ax.set_title(label)
    ax.grid(alpha=0.3)
    if ax is axes[1, 0]:
        ax.legend(fontsize=8)

# Hide empty axis
axes[2, 0].plot(spec.spectral_axis.to_value(u.GHz), spec.data, "k-", lw=1.2)
axes[2, 0].set_title("Spectrum with hanning apodization")

# Shared x-label
axes[2, 0].set_xlabel("OPD (mm)")
axes[1, 1].set_xlabel("OPD (mm)")
