"""Illustration of the three OPD geometries for a top-hat spectrum (100–300 GHz)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / "tests"))

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS

from ftsdata import (
    SingleSidedOpdData,
    DoubleSidedOpdData,
    OneSidedOpdData,
)
from models import rect_itg


# ─── Analytical interferogram of a top-hat spectrum ──────────────────
# Top-hat between 100 and 300 GHz
nu_min = 100  # GHz
nu_max = 300  # GHz
center_wn = (nu_min + nu_max) / 2  # 200 GHz
sigma_wn = (nu_max - nu_min) / 2    # 100 GHz

# OPD sampling
dopd_mm = 0.2  # mm
opd_max_mm = 50  # mm

# ─── 1) Double-sided: symmetric around ZPD (0) ──────────────────────
opd_ds = np.arange(-opd_max_mm, opd_max_mm + dopd_mm, dopd_mm) * u.mm
itg_ds_data = rect_itg(opd_ds.value, center_wn, sigma_wn)[:, np.newaxis, np.newaxis]
itg_ds = SingleSidedOpdData.from_array(opd_ds, itg_ds_data)

# ─── 2) One-sided: ZPD at the start (OPD ≥ 0) ───────────────────────
opd_os = np.arange(0, 2 * opd_max_mm + dopd_mm, dopd_mm) * u.mm
itg_os_data = rect_itg(opd_os.value, center_wn, sigma_wn)[:, np.newaxis, np.newaxis]
itg_os = SingleSidedOpdData.from_array(opd_os, itg_os_data)

# ─── 3) Single-sided (asymmetric): ZPD at 25 % from the start ───────
opd_ss = (np.arange(len(opd_ds)) - 0.25 * len(opd_ds)) * dopd_mm * u.mm
itg_ss_data = rect_itg(opd_ss.value, center_wn, sigma_wn)[:, np.newaxis, np.newaxis]
itg_ss = SingleSidedOpdData.from_array(opd_ss, itg_ss_data)

# ─── Verify class dispatch ──────────────────────────────────────────
assert isinstance(itg_ds, DoubleSidedOpdData), f"Got {type(itg_ds)}"
assert isinstance(itg_os, OneSidedOpdData), f"Got {type(itg_os)}"
assert isinstance(itg_ss, SingleSidedOpdData), f"Got {type(itg_ss)}"

# ─── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex="col", constrained_layout=True)

# Colour per geometry
col_ds = "#1f77b4"
col_os = "#ff7f0e"
col_ss = "#2ca02c"

# Frequency axis for the spectrum
freq = np.linspace(0, 600, 6000)

# ── Top-hat spectrum ──
spectrum = np.where((freq >= 100) & (freq <= 300), 1.0, 0.0)
axes[0, 0].plot(freq, spectrum, "k-", lw=1.5)
axes[0, 0].axvspan(100, 300, color="C0", alpha=0.15)
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_title("Input spectrum (top-hat 100–300 GHz)")
axes[0, 0].set_xlim(0, 600)
axes[0, 0].grid(alpha=0.3)

# ── ILS kernel (for the double-sided OPD sampling) ──
# sinc(x) = sin(pi*x)/(pi*x), first zero at x=1 → Δν = c/(2*OPD_max)
# For OPD_max = 50 mm, Δν ≈ 3 GHz
ils = np.abs(np.sinc(freq * dopd_mm * 2 * opd_max_mm / 299.792458))  # approx
axes[0, 1].plot(freq, ils / np.max(ils), "k-", lw=1.2)
axes[0, 1].set_xlabel("Frequency (GHz)")
axes[0, 1].set_ylabel("Normalised ILS")
axes[0, 1].set_title(f"ILS for OPD_max = {opd_max_mm} mm, ΔOPD = {dopd_mm} mm")
axes[0, 1].set_xlim(0, 400)
axes[0, 1].grid(alpha=0.3)

# ── Interferograms ──
for ax, itg, label, color in [
    (axes[1, 0], itg_ds, "DoubleSidedOpdData\n(symmetric, ZPD centred)", col_ds),
    (axes[1, 1], itg_os, "OneSidedOpdData\n(ZPD at start, OPD ≥ 0)", col_os),
    (axes[2, 0], itg_ss, "SingleSidedOpdData\n(asymmetric, ZPD offset)", col_ss),
]:
    opd = itg.opd_axis.to_value(u.mm)
    data = itg.data[:, 0, 0]  # (M,) from (M, 1, 1)
    ax.plot(opd, data, color=color, lw=1.2)
    ax.axvline(0, color="k", ls=":", lw=0.8, label="ZPD (OPD = 0)")
    ax.set_ylabel("Amplitude")
    ax.set_title(label)
    ax.grid(alpha=0.3)
    if ax is axes[1, 0]:
        ax.legend(fontsize=8)

# Hide empty axis
axes[2, 1].axis("off")

# Shared x-label
axes[2, 0].set_xlabel("OPD (mm)")
axes[1, 1].set_xlabel("OPD (mm)")

fig.suptitle("Interferogram geometries for a 100–300 GHz top-hat spectrum", y=0.98)
fig.savefig(__file__.replace(".py", ".png"), dpi=180)
print(f"Saved to {__file__.replace('.py', '.png')}")