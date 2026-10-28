import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

c_mm_GHz = 299.792458  # mm·GHz

freq0_target = 300.0  # GHz — target; actual freq snapped to nearest bin
dopd = 0.025  # mm

opdmax_arr = np.array([20, 30, 40, 60, 80, 100, 120, 150, 200])
fwhm_boxcar = []
fwhm_hanning = []


def gaussian(x, x0, sigma, A):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


for opd_max in opdmax_arr:
    for use_hanning, store in [(False, fwhm_boxcar), (True, fwhm_hanning)]:
        # Exclude endpoint so N = 2*opd_max/dopd is exact
        opds = np.arange(-opd_max, opd_max, dopd)
        n = len(opds)
        # Snap freq0 to the nearest FFT bin for this N
        freq_bin = c_mm_GHz / (n * dopd)
        k = round(freq0_target / freq_bin)
        freq0_GHz = k * freq_bin
        itg = np.cos(2 * np.pi * freq0_GHz * opds / c_mm_GHz)
        if use_hanning:
            win = np.ones(n)
            half = n // 2
            win[half:] = np.hanning(2 * (n - half))[n - half :]
            itg = itg * win
        nfft = 20 * n
        spec = np.real(np.fft.rfft(itg, n=nfft))
        freq = np.fft.rfftfreq(nfft, d=dopd / c_mm_GHz)
        mask = (freq > freq0_GHz - 30) & (freq < freq0_GHz + 30)
        try:
            popt, _ = curve_fit(
                gaussian,
                freq[mask],
                spec[mask],
                p0=[freq0_GHz, c_mm_GHz / (2 * opd_max) / 2.355, spec[mask].max()],
            )
            store.append(2.355 * abs(popt[1]))
        except Exception:
            store.append(np.nan)

fwhm_boxcar = np.array(fwhm_boxcar)
fwhm_hanning = np.array(fwhm_hanning)
theory = c_mm_GHz / (2 * opdmax_arr)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(opdmax_arr, theory, "k--", lw=1, label=r"$c\,/\,(2\,\mathrm{OPD_{max}})$")
ax.plot(opdmax_arr, fwhm_boxcar, "C0o-", lw=1.5, ms=5, label="Boxcar (fitted)")
ax.plot(opdmax_arr, fwhm_hanning, "C2s-", lw=1.5, ms=5, label="Hanning (fitted)")
ax.set_xlabel(r"OPD$_{\rm max}$ (mm)")
ax.set_ylabel("FWHM (GHz)")
ax.set_title("ILS spectral resolution vs maximum OPD")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
