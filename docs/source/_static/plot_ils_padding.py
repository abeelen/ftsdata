import matplotlib.pyplot as plt
import numpy as np

c_mm_GHz = 299.792458  # mm·GHz

opd_max = 120.0  # mm
dopd = 0.025  # mm

# Build double-sided interferogram (exclude endpoint so N = 2*opd_max/dopd is exact)
opds = np.arange(-opd_max, opd_max, dopd)
n = len(opds)  # 9600

# Choose freq0 as the nearest integer multiple of the FFT bin spacing
# so it lands exactly on a bin regardless of zero-padding factor.
# freq_bin = c / (N * dopd)  =>  freq0 = k * freq_bin  (k integer)
freq_bin = c_mm_GHz / (n * dopd)  # ~ 1.249 GHz
k = round(300.0 / freq_bin)  # 240
freq0_GHz = k * freq_bin  # ~ 299.792 GHz  (exact bin)

itg = np.cos(2 * np.pi * freq0_GHz * opds / c_mm_GHz)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)

configs = [
    ("No padding\n(boxcar)", 1, None, "C0"),
    ("Padding × 10\n(boxcar)", 10, None, "C1"),
    ("Padding × 10\n(Hanning)", 10, "hanning", "C2"),
]

for ax, (label, npad, apod, color) in zip(axes, configs):
    win = np.ones(n)
    if apod == "hanning":
        half = n // 2
        win[half:] = np.hanning(2 * (n - half))[n - half :]

    itg_w = itg * win
    nfft = npad * n
    spec = np.real(np.fft.rfft(itg_w, n=nfft))
    freq = np.fft.rfftfreq(nfft, d=dopd / c_mm_GHz)

    mask = (freq > 250) & (freq < 350)
    peak = spec[mask].max()
    ax.plot(freq[mask], spec[mask] / peak, color=color, lw=1.2)
    ax.axvline(freq0_GHz, color="gray", ls="--", lw=0.8, label="300 GHz")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_title(label)
    ax.set_xlim(275, 325)
    ax.set_ylim(-0.25, 1.1)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Normalised amplitude")
fig.suptitle(
    "ILS for a monochromatic 300 GHz source  |  OPD$_{max}$ = 120 mm",
    y=1.01,
)
fig.tight_layout()
