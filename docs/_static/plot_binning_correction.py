import numpy as np
import matplotlib.pyplot as plt

c_mm_GHz = 299.792458  # mm·GHz

# Gaussian input spectrum centred at 400 GHz, sigma=20 GHz
freq_GHz = np.linspace(50, 500, 5000)
sigma = 20.0
spec_true = np.exp(-0.5 * ((freq_GHz - 400) / sigma) ** 2)

dopd_bins = [0.05, 0.10, 0.20, 0.30]  # mm
colors = ["C0", "C1", "C2", "C3"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for dopd, color in zip(dopd_bins, colors):
    sinc_factor = np.sinc(freq_GHz * dopd / c_mm_GHz)
    spec_binned = spec_true * sinc_factor
    spec_corr = spec_binned / sinc_factor

    nyq = c_mm_GHz / (2 * dopd)
    label = rf"$\Delta$opd = {dopd*1e3:.0f} µm  ($\nu_{{\rm Nyq}}$ = {nyq:.0f} GHz)"
    axes[0].plot(freq_GHz, spec_binned / spec_true.max(), color=color, lw=1.3, label=label)
    axes[1].plot(freq_GHz, np.abs(spec_corr - spec_true) / spec_true.max() * 100, color=color, lw=1.3, label=label)
    axes[0].axvline(nyq, color=color, lw=0.8, ls=":")
    axes[1].axvline(nyq, color=color, lw=0.8, ls=":")

axes[0].plot(freq_GHz, spec_true / spec_true.max(), "k--", lw=1.2, label="True spectrum")
axes[0].set_xlabel("Frequency (GHz)")
axes[0].set_ylabel("Normalised amplitude")
axes[0].set_title("Binned spectrum (no correction)")
axes[0].legend(fontsize=8)
axes[0].set_xlim(50, 500)

axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_ylabel("Residual after correction (%)")
axes[1].set_title("Residual error after sinc correction")
axes[1].set_xlim(50, 500)
axes[1].set_ylim(-0.5, 5)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle(
    "Effect of interferogram binning on a Gaussian line at 400 GHz",
    y=1.01,
)
fig.tight_layout()
