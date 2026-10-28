"""Opacity correction keeps the optional kidsdata ATM integration lazy."""

import numpy as np

from ftsdata import FTSData


class FakeATM:
    """Minimal stand-in for ``kidsdata.forward_model.ATM``."""

    def __init__(self, bins):
        self.bins = bins

    def transmission(self, airmass, pwv):
        return self.bins, np.ones(self.bins.shape)


def test_correct_opacity_accepts_explicit_atm(gaussian_ds):
    spectrum = gaussian_ds.to_spectra()
    corrected = spectrum.correct_opacity(elevation=60, pwv=1.0, atm=FakeATM(spectrum.spectral_bins))

    assert isinstance(corrected, FTSData)
    np.testing.assert_allclose(corrected.data, spectrum.data)
