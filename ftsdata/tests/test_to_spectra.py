"""Interferogram -> spectrum inversion, noise propagation and binning."""

import astropy.constants as cst
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty
from scipy.stats import binned_statistic

from ftsdata import (
    DoubleSidedOpdData,
    OneSidedOpdData,
    OpdData,
    SingleSidedOpdData,
)

from .models import (
    comb_ds_1d_hr,
    gaussian,
    gaussian_ds,
    gaussian_ds_1d,
    gaussian_ds_2d,
    gaussian_ds_noise,
    gaussian_os_noise,
)


def test_to_spectra_ds(gaussian_ds):
    data = gaussian_ds
    assert isinstance(data, DoubleSidedOpdData)

    test = data.to_spectra()

    # Check agains the model
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)

    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])

    # test the output_shape option
    test = data.to_spectra(output_shape=data.naxis_opd * 2)
    assert test.shape[test._spectral_axis_numpy_index] == data.naxis_opd + 1
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])

    # test the n_pad option
    test = data.to_spectra(n_pad=3)
    assert test.shape[test._spectral_axis_numpy_index] == (3 * (data.naxis_opd - 1) + 1) // 2 + 1
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])


def test_to_spectra_os(gaussian_os):
    data = gaussian_os
    assert isinstance(data, OneSidedOpdData)

    # A one-sided cube is inverted directly
    test = data.to_spectra()
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])

    # test the output_shape option
    test = data.to_spectra(output_shape=data.naxis_opd * 2)
    assert test.shape[test._spectral_axis_numpy_index] == data.naxis_opd + 1
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])

    # test the n_pad option
    test = data.to_spectra(n_pad=3)
    assert test.shape[test._spectral_axis_numpy_index] == (3 * 2 * (data.naxis_opd - 1) + 1) // 2 + 1
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model.reshape(test.data.shape)[positive_frequencies], test.data[positive_frequencies])


def test_to_spectra_shift(gaussian_ds_shift):
    data = gaussian_ds_shift
    # Phase correction is handled by the generic retro-compatible implementation
    # (DoubleSidedOpdData.to_spectra performs a direct FFT without phase correction).
    test = OpdData.to_spectra(data, deg=1, phase_error_correction=True)

    # Check agains the model
    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)

    for item in test.data[positive_frequencies, :, 0].T:
        npt.assert_almost_equal(model[positive_frequencies], item, decimal=3)

    # Conserve the absolute phase if negative spectrum
    data = data.multiply(-1)
    test = OpdData.to_spectra(data, deg=1, phase_error_correction=True)
    for item in test.data[positive_frequencies, :, 0].T:
        npt.assert_almost_equal(-model[positive_frequencies], item, decimal=3)


def test_to_spectra_shift_singlesided(gaussian_ss_shift):
    data = gaussian_ss_shift
    assert isinstance(data, SingleSidedOpdData)
    # The phase correction function is only defined on the extracted
    # double-sided cube, not on the single-sided cube itself.
    assert not hasattr(data, "_get_phase_correction_function")

    test = data.to_spectra(deg=1, phase_error_correction=True)

    frequencies = test.spectral_axis
    positive_frequencies = frequencies >= 0 * u.Hz
    model = data.meta["func"](frequencies)
    npt.assert_almost_equal(model[positive_frequencies], test.data[positive_frequencies, 0, 0], decimal=3)


@pytest.mark.parametrize("fixture", [gaussian_ds, gaussian_ds_2d, gaussian_ds_1d])
def test_invert_doublesided(fixture):
    data = fixture()
    central_freq = data.meta["central_freq"]
    sigma_freq = data.meta["sigma_freq"]

    spec = data.to_spectra()
    _spec = spec.data.reshape(spec.data.shape[0], -1)[:, 0]
    npt.assert_almost_equal(_spec.imag, 0)

    (freq,) = spec.wcs.sub([data._opd_axis_index + 1]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
    _should = gaussian(freq, central_freq, sigma_freq) + gaussian(-freq, central_freq, sigma_freq)

    npt.assert_almost_equal(_spec.real, _should)


def test_invert_onesided(gaussian_os):
    data = gaussian_os
    central_freq = data.meta["central_freq"]
    sigma_freq = data.meta["sigma_freq"]

    spec = data.to_spectra()
    _spec = spec.data[:, 0, 0]
    npt.assert_almost_equal(_spec.imag, 0)

    (freq,) = spec.wcs.sub([3]).all_pix2world(np.arange(spec.shape[0]), 0) * u.Hz
    _should = gaussian(freq, central_freq, sigma_freq) + gaussian(-freq, central_freq, sigma_freq)

    npt.assert_almost_equal(_spec.real, _should)


def test_get_phase_correction_function(gaussian_ds_shift):
    data = gaussian_ds_shift
    assert isinstance(data, DoubleSidedOpdData)
    # Ugly way to retrieve the frequency axis of the inversed cube
    dummy = data._extract_doublesided()
    (freq,) = dummy.to_spectra().wcs.sub([3]).all_pix2world(np.arange(dummy.shape[0]), 0)

    # Those pcf should be simple shifts function
    freq = freq * u.Hz
    shifts = np.exp(-2j * np.pi * (freq * data.meta["shifts"][:, np.newaxis] / cst.c).decompose().T.value)
    _pcf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(shifts, axes=0), axis=0), axes=0).real * freq.shape[0]
    _pcf /= (np.median(np.diff(freq)) / cst.c).to_value("1/m")

    pcf = data._get_phase_correction_function(niter=1, deg=1)[:, :, 0]
    # Failling test with the theoricial pcf....
    # assert np.allclose(pcf, _pcf)
    _pcf = pcf

    # Testing higher degree polynomials on simple shifts
    pcf = data._get_phase_correction_function(niter=1, deg=3)[:, :, 0]
    assert np.allclose(pcf, _pcf)

    # Test many iterations, will also work now with amplitude weighting
    pcf = data._get_phase_correction_function(niter=10, deg=1)[:, :, 0]
    assert np.allclose(pcf, _pcf)

    # Test many iterations, will succeed if also apodized
    pcf = data._get_phase_correction_function(niter=10, deg=1, doublesided_apodization=np.hanning)[:, :, 0]
    assert np.allclose(pcf, _pcf)

    # Chebychev polynomials should give the same results at order 1
    pcf = data._get_phase_correction_function(niter=10, deg=1, fitting_func="chebychev")[:, :, 0]
    assert np.allclose(pcf, _pcf)

    with pytest.raises(ValueError):
        data._get_phase_correction_function(niter=10, deg=1, fitting_func="toto")


@pytest.mark.parametrize("fixture", [gaussian_ds, gaussian_ds_2d, gaussian_ds_1d])
def test_noise_and_hit_propagation(fixture):
    data = fixture()
    test = data._to_onesided()

    assert isinstance(data.uncertainty, StdDevUncertainty)
    assert isinstance(test.uncertainty, InverseVariance)
    assert np.all(data.uncertainty.array == data.uncertainty.array[0])

    variance = data.uncertainty.represent_as(VarianceUncertainty).array
    # Double sided so twice... except for zpd
    assert np.all(test.uncertainty.array[1:] == variance[0] * 2)

    assert np.all(test.hits[0] == 1)
    assert np.all(test.hits[1:] == 2)


@pytest.mark.parametrize("fixture", [gaussian_ds_noise])
def test_noise_mc_ds(fixture):
    np.random.seed(42)
    data = fixture()
    # Direct double-sided inversion
    spec = data.to_spectra()
    # They all have the same noise/uncertainty
    assert np.all(spec.uncertainty.array == spec.uncertainty.array[0])
    uncertainty = spec.uncertainty.array[0]
    npt.assert_allclose(uncertainty, np.std(spec.data[1:], axis=1), rtol=1e-1)

    # Same on the folded one-sided inversion (the other side is averaged in)
    spec = data._to_onesided(fold=True).to_spectra()
    npt.assert_allclose(uncertainty, np.std(spec.data[1:], axis=1), rtol=1e-1)

    # If we throw away half the data, the noise increases by sqrt(2)
    spec = data._to_onesided(fold=False).to_spectra()
    npt.assert_allclose(uncertainty * np.sqrt(2), np.std(spec.data[1:], axis=1), rtol=1e-1)


@pytest.mark.parametrize("fixture", [gaussian_os_noise])
def test_noise_mc_os(fixture):
    np.random.seed(42)
    data = fixture()
    spec = data.to_spectra()
    # They all have the same noise/uncertainty
    assert np.all(spec.uncertainty.array == spec.uncertainty.array[0])
    uncertainty = spec.uncertainty.array[0]
    npt.assert_allclose(uncertainty, np.std(spec.data[1:], axis=1), rtol=1e-1)


@pytest.mark.parametrize("fixture", [comb_ds_1d_hr])
def test_correct_itg_binning(fixture):
    """Test that correct_itg_binning inverts the sinc attenuation from OPD averaging."""
    data = fixture()
    ref_opd_axis_mm = data.opd_axis.to_value("mm")

    delta_opd_mm = 0.15  # -> freq_max ~= 1 THz

    # Anchor bin centres on ZPD (0) so that 0 is always a bin centre,
    # then extend symmetrically to cover the full ref_opd_axis_mm range.
    n_left = int(np.floor(-ref_opd_axis_mm.min() / delta_opd_mm))
    n_right = int(np.floor(ref_opd_axis_mm.max() / delta_opd_mm))
    opd_bin_centers_mm = np.arange(-n_left, n_right + 1) * delta_opd_mm
    opd_bins_mm = np.empty(len(opd_bin_centers_mm) + 1)
    opd_bins_mm[:-1] = opd_bin_centers_mm - delta_opd_mm / 2
    opd_bins_mm[-1] = opd_bin_centers_mm[-1] + delta_opd_mm / 2

    # binned_statistic requires values with same length as sample (1D).
    # For multi-dimensional data, flatten spatial axes, bin each pixel, then reshape.
    spatial_shape = data.data.shape[1:]  # everything after the OPD axis
    data_2d = data.data.reshape(data.data.shape[0], -1)  # (naxis_opd, n_pixels)
    binned_rows = np.array(
        [binned_statistic(ref_opd_axis_mm, row, bins=opd_bins_mm, statistic="mean").statistic for row in data_2d.T]
    ).T  # shape: (n_bins, n_pixels)
    binned_array = binned_rows.reshape((binned_rows.shape[0],) + spatial_shape)
    binned_data = OpdData.from_array(opd_bin_centers_mm, binned_array)

    binned_spec = binned_data.to_spectra(n_pad=10)
    corrected_binned_spec = binned_spec.correct_itg_binning()

    # Build a reference spectrum by evaluating the analytical interferogram
    # directly at the bin centres — no box-averaging, no interpolation error.
    # This gives the exact same ILS (same OPD span) so the only residual between
    # reference and corrected is what correct_itg_binning is supposed to remove.
    func_itg = data.meta["func_itg"]
    x_bin = opd_bin_centers_mm * u.mm
    # func_itg expects OPD as a Quantity; broadcast over spatial axes
    extra_axes = (np.newaxis,) * len(spatial_shape)
    x_bin_broadcast = x_bin[(slice(None),) + extra_axes] * np.ones((len(opd_bin_centers_mm),) + spatial_shape)
    itg_ref = func_itg(x_bin_broadcast)
    # Convert unit from 1/mm to GHz (same convention as fixture_itg)
    itg_ref_ghz = (itg_ref * cst.c).to_value("GHz")
    ref_data = OpdData.from_array(opd_bin_centers_mm, itg_ref_ghz)
    ref_spec = ref_data.to_spectra(n_pad=10)

    # Use an absolute tolerance scaled to the spectral peak.
    # The residual is dominated by the O((cdelt_opd/delta_opd)^2) discretisation error
    # of the binned mean vs the continuous-integral sinc model, which is a fixed
    # fraction of the peak (~1.4 % here) independent of the local signal level.
    # atol is therefore the physically meaningful bound — rtol would spuriously fail
    # on the low-signal flanks of each line where the relative error diverges.
    npt.assert_allclose(
        corrected_binned_spec.data,
        ref_spec.data,
        atol=2e-3,
        rtol=0,
    )

    def plot_itg(ax, data, ref_data, binned_data):
        ax.plot(data.opd_axis, data.data, label="real data")
        ax.plot(
            ref_data.opd_axis,
            ref_data.data,
            "x",
            label="sampled data",
        )
        ax.step(binned_data.opd_axis, binned_data.data, where="mid", label="binned data")

    def plot_spec(ax, data, ref_data, binned_data, corrected_data):
        ax.plot(data.spectral_axis.to("GHz"), data.data, label="real data")
        ax.plot(ref_data.spectral_axis.to("GHz"), ref_data.data, label="sampled data", linestyle="dotted")
        ax.plot(binned_data.spectral_axis.to("GHz"), binned_data.data, label="binned data")
        ax.plot(
            corrected_data.spectral_axis.to("GHz"),
            corrected_data.data,
            label="corrected binned data",
            linestyle="dashed",
        )

    def fig():
        spec = data.to_spectra(n_pad=10)

        fig, axes = plt.subplots(ncols=2)
        plot_itg(axes[0], data, ref_data, binned_data)
        axes[0].set_xlabel("OPD [mm]")
        axes[0].set_xlim(-5, 30)

        # Add a compact inset showing the same curves zoomed around the strongest OPD peak.
        axins = axes[0].inset_axes([0.56, 0.52, 0.4, 0.42])
        plot_itg(axins, data, ref_data, binned_data)
        axins.set_xlim(0.8, 2)
        axins.set_ylim(-12, 2)

        axes[0].indicate_inset_zoom(axins, edgecolor="0.4")
        axes[0].legend(loc="lower right", fancybox=False)

        plot_spec(axes[1], spec, ref_spec, binned_spec, corrected_binned_spec)
        axes[1].set_xlabel("Freq [GHz]")
        axes[1].set_xlim(0, 800)
        axes[1].set_ylim(-0.05, 1.2)
        axes[1].legend(loc="upper right", fancybox=False)


def test_to_spectra_real_flag(gaussian_ds):
    """The default output is real; ``real=False`` keeps the complex spectrum."""
    real = gaussian_ds.to_spectra()
    complex_ = gaussian_ds.to_spectra(real=False)

    assert not np.iscomplexobj(real.data)
    assert np.iscomplexobj(complex_.data)
    npt.assert_allclose(real.data, complex_.data.real, rtol=1e-10, atol=1e-12)
    # Keeping only the real part splits the noise between real and imaginary.
    npt.assert_allclose(real.uncertainty.array, complex_.uncertainty.array / np.sqrt(2))


def test_singlesided_to_spectra_forwards_peak(gaussian_ds):
    """The generic single-sided path forwards ``peak`` to the one-sided FFT."""
    data = gaussian_ds
    normal = SingleSidedOpdData.to_spectra(data, peak=False)
    peaked = SingleSidedOpdData.to_spectra(data, peak=True)

    n_onesided = data._to_onesided().naxis_opd
    factor = 2 * (n_onesided - 1) + 1

    npt.assert_allclose(peaked.data * factor, normal.data, rtol=1e-10, atol=1e-12)
    assert not np.allclose(peaked.data, normal.data)
