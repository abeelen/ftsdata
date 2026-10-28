"""Spectrum -> interferogram conversion."""

import numpy as np
import numpy.testing as npt

from ftsdata import OneSidedOpdData, SpecData


def test_to_itg(gaussian_ds):
    spec = gaussian_ds.to_spectra()
    assert isinstance(spec, SpecData)

    itg = spec.to_itg(None, dopd_mm=0.1)
    assert isinstance(itg, OneSidedOpdData)
    assert itg.data.shape[1:] == spec.data.shape[1:]
    assert not np.iscomplexobj(itg.data)
    assert np.all(np.isfinite(itg.data))

    # One-sided OPD axis: starts at zero and is sampled at (about) dopd_mm.
    npt.assert_allclose(itg.opd_axis.to_value("mm")[0], 0, atol=1e-6)
    npt.assert_allclose(np.median(np.diff(itg.opd_axis.to_value("mm"))), 0.1, rtol=1e-3)


def test_to_itg_from_complex_spectrum(gaussian_ds, caplog):
    real = gaussian_ds.to_spectra()
    complex_ = gaussian_ds.to_spectra(real=False)
    assert np.iscomplexobj(complex_.data)

    expected = real.to_itg(None, dopd_mm=0.1)
    with caplog.at_level("WARNING", logger="FTSData"):
        actual = complex_.to_itg(None, dopd_mm=0.1)

    # The imaginary part is discarded, with a warning.
    assert "imaginary part" in caplog.text
    npt.assert_allclose(actual.data, expected.data)
    assert not np.iscomplexobj(actual.data)


def test_to_itg_binning_option(gaussian_ds):
    spec = gaussian_ds.to_spectra()
    corrected = spec.to_itg(None, dopd_mm=0.1, binning=True)
    raw = spec.to_itg(None, dopd_mm=0.1, binning=False)
    assert corrected.data.shape == raw.data.shape
    # The binning correction is a smooth function of frequency, so it only
    # rescales the interferogram; it must not produce NaNs or infinities.
    assert np.all(np.isfinite(corrected.data))
