"""Analytic models and cube factories used by the ftsdata tests.

The helpers evaluate simple spectra and their analytic interferograms, then
build cubes with the right OPD geometry.  The ``partial`` objects at the bottom
of the module are turned into pytest fixtures by ``conftest.py``.
"""

from __future__ import annotations

import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from functools import partial

from ftsdata import OpdData, _classify_opd


def gen_empty_ftsdata(n_itg: int = 1024, n_pix: int | tuple | None = None, itg_type: str = "ds") -> OpdData:
    """Generate a double sided FTSData with random data for testing."""

    if isinstance(n_pix, int):
        n_pix = (n_pix,)

    shape = (n_itg + 1,)
    if n_pix is not None:
        shape += n_pix

    data = np.zeros(shape)
    hits = np.ones_like(data)
    mask = np.zeros_like(data)
    uncertainty = StdDevUncertainty(np.ones_like(data))

    naxis = len(shape)
    wcs = WCS(naxis=naxis)
    if naxis == 1:
        ctype = ["OPD"]
    elif naxis == 2:
        ctype = ["PIX", "OPD"]
    elif naxis == 3:
        ctype = ["OLON-SFL", "OLAT-SFL", "OPD"]
    else:
        raise ValueError("Data shape not supported")

    if itg_type == "ds":
        opd_crpix = n_itg // 2 + 1
    elif itg_type == "ss":
        opd_crpix = n_itg // 4 + 1  # One quarter of the dataset
    elif itg_type == "ss_low":
        opd_crpix = n_itg * 3 // 4 + 1  # 3 quarted of the dataset
    elif itg_type == "os":
        opd_crpix = 1
    else:
        raise ValueError(f"Unknon type {type}")

    wcs.wcs.ctype = ctype
    wcs.wcs.crpix[-1] = opd_crpix
    wcs.wcs.cunit[-1] = "mm"

    cls = _classify_opd(wcs, shape=shape)
    return cls(data, wcs=wcs, hits=hits, mask=mask, uncertainty=uncertainty)


def rect(freq_GHz, center_GHz, sigma_GHz):
    output = (np.abs(freq_GHz - center_GHz) <= sigma_GHz / 2).astype(float)
    return output


def rect_itg(opd_mm, center_wn, sigma_wn):
    shift = np.cos(2 * np.pi * u.Quantity(center_wn * opd_mm).decompose().value)
    rect_ft = np.sinc(u.Quantity(sigma_wn * opd_mm).decompose().value) * np.abs(sigma_wn)
    # Factor 2 comes from the MPI and fourier approximation
    return shift * rect_ft / 2


def gaussian(freq_GHz, center_GHz, sigma_GHz):
    output = np.exp(-((freq_GHz - center_GHz) ** 2) / (2 * sigma_GHz**2))
    return output


def gaussian_itg(opd_mm, center_wn, sigma_wn):
    shift = np.cos(2 * np.pi * u.Quantity(center_wn * opd_mm).decompose().value)
    gaussian_ft = (
        np.exp(-(2 * np.pi**2 * u.Quantity(sigma_wn * opd_mm).decompose().value ** 2)) * np.sqrt(2 * np.pi) * sigma_wn
    )
    # Factor 2 comes from the MPI and fourier approximation
    return shift * gaussian_ft / 2


def comb(freq_GHz, centers_GHz, sigma_GHz):
    return sum(gaussian(freq_GHz, center_GHz, sigma_GHz) for center_GHz in centers_GHz)


def comb_itg(opd_mm, centers_wn, sigma_wn):
    return sum(gaussian_itg(opd_mm, center_wn, sigma_wn) for center_wn in centers_wn)


def blackbody(freq_GHz, temperature_K, scale=1):
    h_over_k = 0.04799243073366221  # GHz / K == (const.h * 1 * u.GHz / (const.k_B * 1 * u.K)).decompose().value
    two_h3_over_c2 = 1474.4994647625417  # Jy / u.sr / GHz**3 == (2 * const.h * (1 * u.GHz)**3 / (const.c**2) / u.sr).to(u.Jy/u.sr).value
    log_boltz = h_over_k * freq_GHz / temperature_K
    boltzm1 = np.expm1(log_boltz)
    bb_nu = two_h3_over_c2 * freq_GHz**3 * scale / boltzm1
    return bb_nu


def diff_blackbody(freq_GHz, temperature1_K, temperature2_K, scale1=1, scale2=1):
    h_over_k = 0.04799243073366221  # GHz / K == (const.h * 1 * u.GHz / (const.k_B * 1 * u.K)).decompose().value
    two_h3_over_c2 = 1474.4994647625417  # Jy / u.sr / GHz**3 == (2 * const.h * (1 * u.GHz)**3 / (const.c**2) / u.sr).to(u.Jy/u.sr).value
    log_boltz1 = h_over_k * freq_GHz / temperature1_K
    log_boltz2 = h_over_k * freq_GHz / temperature2_K
    bb_nu = two_h3_over_c2 * freq_GHz**3 * (scale1 / np.expm1(log_boltz1) - scale2 / np.expm1(log_boltz2))
    return bb_nu


def model_itg(opd_mm, model=None, model_args=None, bandpass=None, df=0.01, max_freq=1e5, **model_kwargs):
    two_pi_over_c = 0.020958450219516818  # GHz mm  == (2 * np.pi * 1 * u.GHz * 1 * u.mm / const.c).decompose().value

    import scipy.integrate as integrate

    if bandpass is None:

        def bandpass(*args):
            return 1.0

    if model_args is None:
        model_args = ()

    def func(nu_GHz):
        return (
            model(nu_GHz, *model_args, **model_kwargs)
            * bandpass(nu_GHz)
            * np.cos(nu_GHz * two_pi_over_c * np.atleast_1d(opd_mm)[:, None])
        )  # noqa: E731

    min_freq = 1e-16
    nu_GHz = np.linspace(min_freq, max_freq, int((max_freq - min_freq) / df))
    itg = integrate.trapezoid(func(nu_GHz), x=nu_GHz, axis=-1)

    # Put back dx units and transform it to 1/mm like the other function
    itg *= (1 * u.GHz / cst.c).to(u.mm**-1)
    # Factor 2 comes from the MPI and fourier approximation
    return itg / 2


def ils(freq, opd_max):
    r"""Compute the normalized ILS.

    Parameters
    ----------
    freq : astropy.units.Quantity
        the frequency axis
    opd_max : astropy.units.Quantity
        the maximum optical path difference

    Returns
    -------
    array_like
        the ILS to convolve the corresponding spectra

    Notes
    -----
    .. math:: FT( rect(a x) / a ) = sinc\left(\frac{\xi]{a}\right)

    with :math:`a = \frac{1}{2 opd_max}` and :math:`\sinc(x) = \frac{\sin(\pi x)}{\pi x}`
    """
    # Normalized to one rectangle function
    wn = freq / cst.c
    return np.sinc(u.Quantity(wn * 2 * opd_max).decompose().value)


def fixture_itg(
    central_freq=120 * u.GHz,
    sigma_freq=10 * u.GHz,
    n_itg=2048,
    n_pix=1,
    shifts=None,
    noise=None,
    cdelt_opd=0.5 * u.mm,
    func_itg=gaussian_itg,
    func=gaussian,
    itg_type="ds",
    ndim=3,
):
    """Generate a FTSData with given integral of a function

    Parameters
    ----------
    central_freq : astropy.units.Quantity
        Central frequency of the function
    sigma_freq : astropy.units.Quantity
        Width of the function
    n_itg : int
        Number of integration points (OPD axis)
    n_pix : int
        Number of pixels in spatial axis
    shifts : array_like or None
        Shift to add to each pixel in mm
    noise : float or None
        Standard deviation of the noise to add to the data
    cdelt_opd : astropy.units.Quantity
        Step size in OPD axis
    func_itg : function
        Function to compute the integral
    func : function
        Function to compute the spectrum
    itg_type : str
        Type of integral ('ds', 'os', 'ss', 'ss_low')
    ndim : int
        Number of dimensions (1, 2 or 3)

    Returns
    -------
    OpdData
        The generated OpdData
    """
    gen_kwargs = dict(n_itg=n_itg, itg_type=itg_type)
    gen_kwargs["n_pix"] = [None, n_pix, (n_pix, n_pix)][ndim - 1]

    data = gen_empty_ftsdata(**gen_kwargs)
    data.wcs.wcs.cdelt[data._opd_axis_index] = cdelt_opd.to(data.wcs.wcs.cunit[data._opd_axis_index]).value

    # temporary fix :
    # data.wcs.wcs.ctype[2] = "opd"

    opd_wcs = data.wcs.sub([data._opd_axis_index + 1])
    cdelt_opd = opd_wcs.wcs.cdelt[0]
    cunit_opd = u.Unit(opd_wcs.wcs.cunit[0])
    naxis_opd = data.shape[0]

    shape = data.shape

    # OPD
    opd_pix = np.repeat(np.arange(naxis_opd), np.prod(shape[1:]))
    (x,) = opd_wcs.all_pix2world(opd_pix, 0) * cunit_opd
    x = x.reshape(shape)

    # Add shift in x
    if shifts is not None:
        x += shifts[np.newaxis, :, np.newaxis]

    center_x = (central_freq / cst.c).decompose()
    sigma_x = (sigma_freq / cst.c).decompose()

    # Should actually be in the fft part
    # norm = (2 * cdelt_opd * cunit_opd).decompose()
    itg = func_itg(x, center_x, sigma_x)  # * norm

    # Convert unit from 1/mm to GHz
    data.data[:] = (itg * cst.c).to_value("GHz")

    if noise is not None:
        assert isinstance(data.uncertainty, StdDevUncertainty)
        data.uncertainty.array[:] = noise
        # Electronic noise
        data_noise = np.random.normal(scale=noise, size=data.shape)
        # # Photon noise # One need the absolute background to get the photon noise, here we just scale with the signal, that does not work well...
        # min_data = data.data.min(axis=0)
        # max_data = data.data.max(axis=0)
        # data_noise = (data.data - min_data) / (max_data - min_data)
        # data_noise *= noise
        # data_noise = np.random.normal(scale=data_noise)
        data.data[:] += data_noise

    data.meta["central_freq"] = central_freq
    data.meta["sigma_freq"] = sigma_freq
    data.meta["shifts"] = shifts
    data.meta["noise"] = noise
    data.meta["func"] = lambda freq: func(freq, central_freq, sigma_freq)
    data.meta["func_itg"] = lambda opd, **kw: func_itg(
        opd, (central_freq / cst.c).decompose(), (sigma_freq / cst.c).decompose(), **kw
    )

    return data


# Define all tests as functions to all easy debugging
fixture_kwargs = dict(func_itg=gaussian_itg, func=gaussian)
gaussian_ds = partial(fixture_itg, **fixture_kwargs, itg_type="ds", ndim=3)
gaussian_ds_2d = partial(fixture_itg, **fixture_kwargs, itg_type="ds", ndim=2)
gaussian_ds_1d = partial(fixture_itg, **fixture_kwargs, itg_type="ds", ndim=1)

gaussian_os = partial(fixture_itg, **fixture_kwargs, itg_type="os")
gaussian_ss = partial(fixture_itg, **fixture_kwargs, itg_type="ss")
gaussian_ss_low = partial(fixture_itg, **fixture_kwargs, itg_type="ss_low")

gaussian_ds_noise = partial(fixture_itg, **fixture_kwargs, itg_type="ds", ndim=2, n_pix=1024, noise=1)
gaussian_os_noise = partial(fixture_itg, **fixture_kwargs, itg_type="os", ndim=2, n_pix=1024, noise=1)

fixture_kwargs = dict(func_itg=rect_itg, func=rect, ndim=1)
rect_ds = partial(fixture_itg, **fixture_kwargs, itg_type="ds")
rect_os = partial(fixture_itg, **fixture_kwargs, itg_type="os")
rect_ss = partial(fixture_itg, **fixture_kwargs, itg_type="ss")

fixture_kwargs = dict(n_pix=3, cdelt_opd=0.3 * u.mm, shifts=[0, 0.15, 0.3] * u.mm)
gaussian_ds_shift = partial(fixture_itg, **fixture_kwargs, func_itg=gaussian_itg, func=gaussian, itg_type="ds")
gaussian_ss_shift = partial(fixture_itg, **fixture_kwargs, func_itg=gaussian_itg, func=gaussian, itg_type="ss")
rect_ds_shift = partial(fixture_itg, **fixture_kwargs, func_itg=rect_itg, func=rect, itg_type="ds")
rect_ss_shift = partial(fixture_itg, **fixture_kwargs, func_itg=rect_itg, func=rect, itg_type="ss")

fixture_kwargs = dict(
    func_itg=comb_itg,
    func=comb,
    cdelt_opd=1e-3 * u.mm,
    n_itg=2**17,
    central_freq=[100, 200, 300, 400, 500, 600, 700] * u.GHz,
    sigma_freq=10 * u.GHz,
)
comb_ds_1d_hr = partial(fixture_itg, **fixture_kwargs, itg_type="ds", ndim=1)
