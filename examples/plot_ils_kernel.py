r"""
Instrumental line shape and spectral resolution
===============================================

The instrumental line shape (ILS) is the response of the FTS to a
monochromatic line.  For a boxcar window it is a sinc whose main-lobe width is
set by the maximum optical path difference:

.. math::

   \mathrm{FWHM} \approx \frac{c}{2\,\mathrm{OPD_{max}}}.

This example compares the ILS for several OPD ranges and shows how Hann
apodization broadens the main lobe.
"""

# %%
# Compute the ILS kernels
# -----------------------
# :func:`~ftsdata.get_ils_kernel` returns the ILS resampled on a regular
# frequency grid, ready to be used as an :class:`astropy.convolution.Kernel`.
import numpy as np
import matplotlib.pyplot as plt

from ftsdata import get_ils_kernel

C_MM_GHZ = 299.792458  # speed of light, in mm GHz
dfreq_GHz = 0.25

# %%
# Plot the normalised kernels as a function of the frequency offset from the
# line centre.
fig, ax = plt.subplots(figsize=(7, 3.5))

for opd_max in (30.0, 60.0, 120.0):
    kernel = get_ils_kernel(opdmax_mm=opd_max, dfreq_GHz=dfreq_GHz)
    offset = (np.arange(kernel.array.size) - (kernel.array.size - 1) // 2) * dfreq_GHz
    fwhm = C_MM_GHZ / (2 * opd_max)
    label = f"OPD$_{{\\max}}$ = {opd_max:.0f} mm (FWHM ≈ {fwhm:.2f} GHz)"
    ax.plot(offset, kernel.array / kernel.array.max(), lw=1.5, label=label)

# Apodization broadens the main lobe and suppresses the sidelobes.
kernel_hann = get_ils_kernel(opdmax_mm=120.0, dfreq_GHz=dfreq_GHz, onesided_apodization=np.hanning)
offset = (np.arange(kernel_hann.array.size) - (kernel_hann.array.size - 1) // 2) * dfreq_GHz
ax.plot(offset, kernel_hann.array / kernel_hann.array.max(), "k--", lw=1.2, label="OPD$_{\\max}$ = 120 mm, Hann")

ax.set_xlim(-6, 6)
ax.set_xlabel("Frequency offset (GHz)")
ax.set_ylabel("Normalised response")
ax.set_title("Instrumental line shape vs maximum OPD")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
