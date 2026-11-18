"""
From spectrum to interferogram and back
=======================================

This example demonstrates the complete FTS workflow: loading spectral data,
applying a smooth bandpass function, converting to interferogram space, and
converting back to spectrum with apodization.

We use a real FITS data cube (50×50 spatial pixels, 181 spectral channels)
from CONCERTO observations (125–305 GHz) and apply a basic smooth tophat
bandpass. At each major step, we show both spatial maps and spectral traces
to understand the data transformations.

Key points illustrated:

- The bandpass limits the spectral range used for the ITG transform
- Converting to ITG and back produces a spectrum covering 0 to Nyquist
  frequency (:math:`c / 2\\Delta\\mathrm{OPD}`), which is much wider than
  the original bandpass
- Apodization (Hann window) reduces spectral sidelobes at the cost of
  resolution
- Zero-padding (:code:`n_pad`) oversamples the recovered spectrum for better
  visualization of line shapes
"""

# %%
# Load the spectral data
# ----------------------
# Start by loading a real FITS cube containing spectral data with
# (frequency, y, x) dimensions.
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from ftsdata import SpecData

# Load the example FITS file
# sphinx-gallery may not define __file__, fall back to relative path
try:
    fits_path = Path(__file__).parent / "data" / "side_spec.fits"
except NameError:
    fits_path = Path("data/side_spec.fits")
freq_cube = SpecData.read(fits_path, format="fits")

print(f"Loaded cube shape: {freq_cube.data.shape}")
print(f"Spectral range: {freq_cube.spectral_axis[0].to(u.GHz):.1f} to {freq_cube.spectral_axis[-1].to(u.GHz):.1f}")
print(f"Channel spacing: {np.median(np.diff(freq_cube.spectral_axis)).to(u.GHz):.3f}")

# Define bright source locations (valid coordinates for this 50×50 cube)
x_sources = [29, 12, 42, 16, 32, 25]
y_sources = [37, 43, 7, 20, 21, 25]
source_names = [f"Source {i+1}" for i in range(len(x_sources))]

# Select one source for detailed spectral comparison
source_idx = 1
x_center, y_center = x_sources[source_idx], y_sources[source_idx]
freq_idx_center = len(freq_cube.spectral_axis) // 2

# %%
# Define and apply a smooth tophat bandpass
# -----------------------------------------
# Create a smooth tophat function that gradually transitions from 0 to 1
# using a Gaussian smoothing kernel. This simulates a typical instrumental
# bandpass response.

def smooth_tophat(freq_GHz, freq_min_GHz=175, freq_max_GHz=250, gaussian_width_GHz=2.0):
    """
    Create a smooth tophat function convolved with a Gaussian.

    Parameters
    ----------
    freq_GHz : array-like
        Frequency values in GHz
    freq_min_GHz : float
        Minimum frequency of the tophat (in GHz)
    freq_max_GHz : float
        Maximum frequency of the tophat (in GHz)
    gaussian_width_GHz : float
        Width (sigma) of the Gaussian smoothing kernel (in GHz)

    Returns
    -------
    response : ndarray
        Bandpass response (0 to 1)
    """
    # Create a tophat function
    tophat = np.zeros_like(freq_GHz)
    mask = (freq_GHz >= freq_min_GHz) & (freq_GHz <= freq_max_GHz)
    tophat[mask] = 1.0

    # Smooth with Gaussian
    # Convert Gaussian width from GHz to pixel units
    freq_res = np.abs(np.median(np.diff(freq_GHz)))
    sigma_pixels = gaussian_width_GHz / freq_res
    response = gaussian_filter1d(tophat, sigma=sigma_pixels, mode="constant", cval=0.0)

    response += 1e-12  # Avoid exact zeros to prevent division issues

    return response


# Apply the bandpass to the cube
freq_values = freq_cube.spectral_axis.to_value("GHz")
bandpass = smooth_tophat(freq_values)

# %%
# Inspect the input spectrum and map
# ============================================
# Plot both the spatial map (at a central frequency) and the spectrum
# at a bright source location. We examine several known bright sources
# to demonstrate the workflow on real astronomical signals.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.5]})

# Left: spatial map at central frequency with source locations marked
ax = axes[0]
im = ax.imshow(freq_cube.data[freq_idx_center, :, :], cmap="viridis", origin="lower")
for i, (x, y) in enumerate(zip(x_sources, y_sources)):
    if i == source_idx:
        ax.scatter(x, y, c="r", marker="+", s=100, lw=2, label=source_names[i])
    else:
        ax.scatter(x, y, marker="+", s=80, lw=1.5, label=source_names[i])
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.set_title(f"Input map at {freq_cube.spectral_axis[freq_idx_center].to(u.GHz):.1f} GHz")
ax.legend(loc="upper right", fontsize=8)

# Right: spectrum at source positions
ax = axes[1]
freq_axis = freq_cube.spectral_axis.to_value("GHz")
for i, (x, y) in enumerate(zip(x_sources, y_sources)):
    if i == source_idx:
        ax.semilogy(freq_axis, freq_cube.data[:, y, x], c="r", lw=2, label=source_names[i])
    else:
        ax.semilogy(freq_axis, freq_cube.data[:, y, x], lw=1, alpha=0.7, label=source_names[i])
ax.axvline(freq_cube.spectral_axis[freq_idx_center].to_value("GHz"), color="red", ls="--", alpha=0.5, label="Map frequency")
ax.plot(freq_axis, bandpass, linestyle="-.", c="k", lw=2, label="Smooth tophat bandpass")
ax.set_ylim(1e-3, 1.1)
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (MJy/sr)")
ax.set_title("Input spectra at source positions")
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Original Spectral Data Cube", fontsize=14)
fig.tight_layout()

# %%
# Convert to interferogram
# =================================
# Transform the filtered spectral data to optical path difference domain
# using inverse FFT. The bandpass function defines the spectral range
# used for the transform.

# dopd_mm = 0.05 mm gives Nyquist frequency ~3000 GHz (well above our 305 GHz)
itg_cube = freq_cube.to_itg(bandpass=smooth_tophat, dopd_mm=0.05)

# Initialize the mask properly for the interferogram
itg_cube.mask = np.zeros_like(itg_cube.data, dtype=bool)

print(f"\nInterferogram shape: {itg_cube.data.shape}")
print(f"OPD range: {itg_cube.opd_axis[0]:.2f} to {itg_cube.opd_axis[-1]:.2f}")
print(f"OPD sampling: {np.diff(itg_cube.opd_axis[:2])[0]:.4f}")
print(f"Nyquist frequency: {(cst.c / (2 * (itg_cube.opd_axis[1] - itg_cube.opd_axis[0]))).to(u.GHz):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.5]})

# Left: Interferogram map at central OPD (zero path difference)
ax = axes[0]
im = ax.imshow(itg_cube.data[0, :, :], cmap="plasma", origin="lower")
for i, (x, y) in enumerate(zip(x_sources, y_sources)):
    if i == source_idx:
        ax.scatter(x, y, c="r", marker="+", s=100, lw=2, label=source_names[i])
    else:
        ax.scatter(x, y, marker="+", s=80, lw=1.5, label=source_names[i])
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.set_title("Interferogram map at OPD = 0 mm")
ax.legend(loc="upper right", fontsize=8)

# Right: Interferogram traces at bright sources
ax = axes[1]
for i, (x, y) in enumerate(zip(x_sources, y_sources)):
    ax.plot(itg_cube.opd_axis.to_value("mm"), itg_cube.data[:, y, x], lw=1.5, alpha=0.8, label=source_names[i])
ax.set_ylim(-2, 2)
ax.set_xlim(0, 10)
ax.set_xlabel("Optical Path Difference (mm)")
ax.set_ylabel("Amplitude")
ax.set_title("Interferograms at sources (zoomed to 10 mm)")
ax.grid(alpha=0.3)
ax.legend(ncol=3, fontsize=8)

fig.suptitle("One-Sided Interferograms", fontsize=14)
fig.tight_layout()

# %%
# Convert back to spectrum
# =================================
# Invert the interferogram using FFT with Hann apodization to demonstrate
# the round-trip transformation and show the effect of apodization.

# Without padding - native resolution
spec_from_itg = itg_cube.to_spectra(apodization=np.hanning)
# With n_pad=4 - 4× oversampled in frequency for better line shape visualization
spec_from_itg_paded = itg_cube.to_spectra(apodization=np.hanning, n_pad=4)

print(f"\nRecovered spectrum shape (native): {spec_from_itg.data.shape}")
print(f"Recovered spectral range: {spec_from_itg.spectral_axis[0].to(u.GHz):.1f} to {spec_from_itg.spectral_axis[-1].to(u.GHz):.1f}")
print(f"Recovered channel spacing: {np.median(np.diff(spec_from_itg.spectral_axis)).to(u.GHz):.3f}")

# The recovered spectrum covers 0 to Nyquist (~3000 GHz), much wider than
# the original 125–305 GHz. To compare with original, we must:
# 1. Divide by the bandpass to correct for the instrumental response
# 2. Slice to the original frequency range

# Bandpass on the recovered frequency grid
bandpass_native = smooth_tophat(spec_from_itg.spectral_axis.to_value("GHz"))
bandpass_paded = smooth_tophat(spec_from_itg_paded.spectral_axis.to_value("GHz"))

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    spec_from_itg_corrected = spec_from_itg.divide(bandpass_native[:, None, None])
    spec_from_itg_paded_corrected = spec_from_itg_paded.divide(bandpass_paded[:, None, None])


# %%
# Compare original and recovered spectra
# ===============================================
# Plot the spectral traces at the source positions, comparing the original
# with the recovered (with and without zero-padding).

fig, axes = plt.subplots(len(x_sources), 1, figsize=(7, 2.5 * len(x_sources)), sharex=True)
if len(x_sources) == 1:
    axes = [axes]

for ax, x, y, name in zip(axes, x_sources, y_sources, source_names):
    ax.semilogy(
        freq_cube.spectral_axis.to_value("GHz"),
        freq_cube.data[:, y, x],
        label="Original",
        lw=2, c="k"
    )
    ax.semilogy(
        spec_from_itg_corrected.spectral_axis.to_value("GHz"),
        spec_from_itg_corrected.data[:, y, x],
        label="Recovered (native)",
        lw=1.5,
        ls="--",
        c="C0",
    )
    ax.semilogy(
        spec_from_itg_paded_corrected.spectral_axis.to_value("GHz"),
        spec_from_itg_paded_corrected.data[:, y, x],
        label="Recovered (4× padded)",
        lw=1,
        ls=":",
        c="C1",
    )
    freq_range = (freq_cube.spectral_axis.to_value("GHz") > 175) & (freq_cube.spectral_axis.to_value("GHz") < 250)
    ax.set_ylim(freq_cube.data[freq_range, y, x].min(), freq_cube.data[freq_range, y, x].max())
    ax.set_ylabel("Flux (MJy/sr)")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_xlim(175 - 5, 250 + 5)
axes[-1].set_xlabel("Frequency (GHz)")
fig.suptitle("Round-trip Comparison (Original vs Recovered)", fontsize=14)
fig.tight_layout()

# %%
# Compare spatial maps at selected frequencies
# =====================================================
# Show a grid of spatial maps at several frequencies across the bandpass,
# comparing original and recovered data.

nX = 7
# Pick nX evenly spaced frequencies across the bandpass
freq_indices = np.linspace(0, len(freq_cube.spectral_axis) - 1, nX, dtype=int)
frequencies = freq_cube.spectral_axis[freq_indices]

fig = plt.figure(figsize=(nX * 2, 4.5), constrained_layout=True)
gs = fig.add_gridspec(2, nX, hspace=0.1, wspace=0.1)

# Determine common color scale
vmin, vmax = 0.01, 0.1
kwargs = dict(origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)

# Top row: Original maps
for i, freq in enumerate(frequencies):
    idx = np.argmin(np.abs(freq_cube.spectral_axis - freq))
    ax = fig.add_subplot(gs[0, i])
    ax.set_axis_off()
    ax.set_title(f"{freq.to_value('GHz'):.0f} GHz", fontsize=9)
    ax.imshow(freq_cube.data[idx, :, :], **kwargs)

# Bottom row: Recovered maps (padded, for better sampling)
for i, freq in enumerate(frequencies):
    idx = np.argmin(np.abs(spec_from_itg_paded.spectral_axis - freq))
    ax = fig.add_subplot(gs[1, i])
    ax.set_axis_off()
    if i == 0:
        ax.set_ylabel("Recovered\n(4× padded)", fontsize=10, rotation=90, labelpad=10)
    ax.imshow(spec_from_itg_paded_corrected.data[idx, :, :], **kwargs)

fig.suptitle("Spatial Map Comparison at Selected Frequencies", fontsize=14, y=1.02)
# constrained_layout handles the layout, no tight_layout needed

plt.show()