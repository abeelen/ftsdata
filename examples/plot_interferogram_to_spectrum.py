"""
From interferogram to spectrum
==============================

This example builds a synthetic double-sided interferogram cube and inverts
it with :meth:`~ftsdata.DoubleSidedOpdData.to_spectra`.

The source is a pair of Gaussian lines near 300 and 350 GHz, observed with a
CONCERTO-like FTS: the optical path difference (OPD) is sampled every 100 µm
between -50 and +50 mm.
"""

# %%
# Build the synthetic interferogram
# ---------------------------------
# The analytic interferogram of a Gaussian line of width :math:`\sigma_\nu`
# centred at :math:`\nu_0` is
#
# .. math::
#
#    I(\tau) = \exp\!\left(-2\pi^2 \sigma_\nu^2 \tau^2\right)
#              \cos(2\pi \nu_0 \tau),
#
# where :math:`\tau = \mathrm{OPD}/c` is expressed in ns.  Summing two lines
# produces the beat pattern visible in the interferogram.
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst

from ftsdata import DoubleSidedOpdData

C_MM_GHZ = cst.c.to_value('mm GHz')

delta_opd_mm = 0.01
opd = np.arange(-25.0, 25.0 + delta_opd_mm/2, delta_opd_mm) * u.mm
tau_ns = opd.to_value(u.mm) / C_MM_GHZ

data = np.zeros_like(tau_ns)
for freq_GHz, sigma_GHz, amplitude in [(300.0, 2.0, 1.0), (350.0, 4.0, 0.6)]:
    envelope = np.exp(-2 * np.pi**2 * sigma_GHz**2 * tau_ns**2)
    data += amplitude * envelope * np.cos(2 * np.pi * freq_GHz * tau_ns)

itg = DoubleSidedOpdData.from_array(opd, data)
print(type(itg).__name__, itg.data.shape)

# %%
# Inspect the interferogram
# -------------------------
# The data axis is the OPD, with the zero path difference (ZPD) at the centre
# of a symmetric double-sided scan.
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(itg.opd_axis.to_value(u.mm), itg.data, lw=0.8)
ax.axvline(0, color="k", ls=":", lw=0.8, label="ZPD")
ax.set_xlabel("OPD (mm)")
ax.set_ylabel("Amplitude")
ax.set_title("Synthetic double-sided interferogram")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

# %%
# Invert the interferogram
# ------------------------
# ``n_pad=10`` zero-pads the interferogram before the FFT.  This oversamples
# the spectrum so that the line shape is resolved; it does not add spectral
# information.  Apodization trades spectral resolution for lower sidelobes.
spec_raw = itg.to_spectra()
spec_raw_hann = itg.to_spectra(apodization=np.hanning)
spec_pad = itg.to_spectra(n_pad=10)
spec_pad_hann = itg.to_spectra(apodization=np.hanning, n_pad=10)

fig, axes = plt.subplots(nrows=2, figsize=(7, 7), sharex=True)
for spec, label in zip([spec_raw, spec_raw_hann], ['raw spectrum', 'w. Hann apodization']):
    axes[0].plot(spec.spectral_axis.to_value('GHz'), spec.data, label=label)
    axes[0].set_ylabel('Amplitude')
for spec, label in zip([spec_pad, spec_pad_hann], ['padded spectrum', 'w. Hann apodization']):
    axes[1].plot(spec.spectral_axis.to_value('GHz'), spec.data, label=label)
    axes[1].set_ylabel('Amplitude')
axes[1].set_xlim(250, 400)
axes[1].set_xlabel("Frequency [GHz]")
fig.suptitle("Spectrum after inversion")
axes[0].legend()
axes[1].legend()
fig.tight_layout()
