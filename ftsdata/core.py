"""Data containers for FTS interferograms and spectra.

The class hierarchy is:

    SpecData
    SingleSidedOpdData (base for all OPD cubes)
      ├── DoubleSidedOpdData
      └── OneSidedOpdData
    FTSData(SingleSidedOpdData, SpecData)

I/O (FITS/HDF5) lives in a separate module and is wired up later.
"""

import logging
import warnings
from copy import deepcopy
from functools import partial

import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.io.fits.verify import VerifyWarning
from astropy.nddata import InverseVariance, NDDataArray, StdDevUncertainty, VarianceUncertainty
from astropy.wcs import WCS
from scipy.signal import fftconvolve, medfilt, windows
from scipy.stats import binned_statistic

try:  # NumPy >= 2.0
    from numpy.exceptions import RankWarning
except ImportError:  # NumPy < 2.0
    from numpy import RankWarning

from ._logging import logged
from ._utils import _insert_in_metadata_fits_safe, elevation_to_airmass
from .mixins import (
    HDUMixinClass,
    HitsMixinClass,
    IdiaMixinClass,
    OpdAxisMixinClass,
    SpectralAxisMixinClass,
    WeightMixinClass,
)
from .transforms import (
    _to_spectra_doublesided_block,
    _to_spectra_onesided_block,
    apply_blockwise,
    get_itg_binning_correction,
)

logger = logging.getLogger("FTSData")


def _get_atm(atm=None):
    """Return the atmospheric transmission model used by opacity correction.

    ``kidsdata`` is an optional dependency: importing it lazily keeps
    ``ftsdata`` importable on its own while preserving the old integration
    when kidsdata is installed.  A compatible model can also be passed
    explicitly through the ``atm`` argument.
    """
    if atm is not None:
        return atm

    try:
        from kidsdata.forward_model import ATM
    except ImportError as exc:
        raise ImportError(
            "correct_opacity requires the optional 'kidsdata' package "
            "(kidsdata.forward_model.ATM). Install kidsdata, or pass a "
            "compatible model with the 'atm' argument."
        ) from exc

    return ATM


@logged
class SpecData(SpectralAxisMixinClass, WeightMixinClass, HDUMixinClass, IdiaMixinClass, HitsMixinClass, NDDataArray):
    """Class to handle spectral cube from FTS data.

    Parameters
    ----------
    data : `~numpy.ndarray` or `FTSData`
        The actual data contained in this `FTSData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.

    uncertainty : `~astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.data.wcs.world_axis_physical_types


    mask : `~numpy.ndarray`-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.

    hits : `~numpy.ndarray`-like, optional
        Hit map for the data, given as a int Numpy array or any object that
        can be converted to a int Numpy array with a shape
        matching that of the data.

    flags : `~numpy.ndarray`-like or `~astropy.nddata.FlagCollection`, optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type (or an object which can be converted
        to a Numpy array) with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.

    wcs : `~astropy.wcs.WCS`, optional
        WCS-object containing the world coordinate system for the data.

    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    unit : `~astropy.units.UnitBase` instance or str, optional
        The units of the data.

    """

    def correct_opacity(self, elevation=None, pwv=None, atm=None):
        """
        Apply correction of opacity based on ATM model for elevation and PWV

        Parameters
        ----------
        elevation : array
        pwv : array
        atm : optional
            Atmospheric model exposing a ``transmission(airmass, pwv)`` method.
            Defaults to ``kidsdata.forward_model.ATM`` when kidsdata is
            installed.

        Returns
        -------
        spec_opacity_corrected : FTSData
            An opacity corrected spectral cube

        Notes
        -----
        elevation might be taken as ::

            elev = kd.get_telescope_data('elevation')
            elevation = np.nanmedian(elev[0])

        pwv might be taken as ::

            pwv_ = kd.get_telescope_data('PWV')
            pwv = np.nanmedian(pwv_[0])

        """
        if elevation is None:
            elevation = self.meta["MED_ELEVATION"]
        assert elevation is not None, "Please provide an elevation"

        if pwv is None:
            pwv = self.meta["MED_PWV"]
        assert pwv is not None, "Please provide a PWV"

        airmass = elevation_to_airmass(elevation, formula="PlaneParallel")

        atm = _get_atm(atm)
        freq_atm, transmission = atm.transmission(airmass, pwv)
        freq_atm = freq_atm.value

        freq_bins = self.spectral_bins

        # Getting the opacity at the sampling of freq_ghz (from cube):
        # delta_freq = np.mean(np.diff(freq_ghz))  # because of non-uniform differences
        # freq_bins = np.zeros(freq_ghz.shape[0] + 1)
        # freq_bins[:-1] = freq_ghz - delta_freq / 2
        # freq_bins[-1] = freq_ghz[-1]

        averaged_transmission, bin_edges, binnumber = binned_statistic(
            freq_atm, transmission, bins=freq_bins, statistic="mean"
        )

        spec_opacity_corrected = self.data.copy() / averaged_transmission[:, None, None]
        fts_data = FTSData(
            spec_opacity_corrected,
            wcs=self.wcs,
            mask=self.mask,
            meta=self.meta,
            hits=self.hits,
            uncertainty=self.uncertainty,
        )

        return fts_data

    def correct_itg_binning(self, delta_opd=None):
        """Correct for the spectral distortion induced by averaging interferogram bins.

        When interferogram data are averaged (binned) over a finite OPD bin width
        ``delta_opd``, the Fourier transform relationship implies that the resulting
        spectrum is multiplied by a sinc function:

        .. math::

            S_{\\mathrm{meas}}(\\nu) = S_{\\mathrm{true}}(\\nu) \\times
            \\operatorname{sinc}\\!\\left(\\frac{\\nu\\,\\Delta\\mathrm{opd}}{c}\\right)

        This method divides the spectrum by that factor to recover
        :math:`S_{\\mathrm{true}}`.

        Parameters
        ----------
        delta_opd : `~astropy.units.Quantity` or float, optional
            Width of the OPD bin over which the interferogram was averaged.
            If a plain float is given it is interpreted as millimetres.
            If not provided, the value stored under ``meta['dopd']`` (set
            automatically by `to_spectra`) is used.

        Returns
        -------
        spec_corrected : SpecData
            A new spectral cube with the binning correction applied.

        """
        if delta_opd is None:
            dopd_str = self.meta.get("dopd", None)
            assert dopd_str is not None, (
                "delta_opd not provided and 'dopd' not found in meta. "
                "Pass delta_opd explicitly or call to_spectra() first."
            )
            delta_opd = u.Quantity(dopd_str)

        if not isinstance(delta_opd, u.Quantity):
            delta_opd = delta_opd * u.mm

        correction = get_itg_binning_correction(self.spectral_axis, delta_opd)

        # Reshape to broadcast over all non-spectral axes
        shape = [1] * self.data.ndim
        shape[self._spectral_axis_numpy_index] = -1
        correction = correction.reshape(shape)

        corrected_data = self.data / correction

        uncertainty = None
        if self.uncertainty is not None:
            unc = self.uncertainty.represent_as(StdDevUncertainty)
            uncertainty = StdDevUncertainty(unc.array / correction)

        return self.__class__(
            corrected_data,
            wcs=self.wcs,
            mask=self.mask,
            meta=self.meta.copy(),
            hits=self.hits,
            uncertainty=uncertainty,
            unit=self.unit,
        )

@logged
class SingleSidedOpdData(OpdAxisMixinClass, WeightMixinClass, HDUMixinClass, IdiaMixinClass, HitsMixinClass, NDDataArray):
    """Class to handle OPD cube of FTS cubes.

    This is the shared base class for interferogram cubes.  Three
    geometry-specific subclasses implement the actual inversion:

    - :class:`DoubleSidedOpdData` for a symmetric OPD axis with the ZPD centred;
    - :class:`OneSidedOpdData` for positive OPDs with the ZPD at the start;
    - :class:`SingleSidedOpdData` for asymmetric OPD axes.

    :meth:`from_array` and the FITS/HDF5 readers dispatch to the appropriate
    subclass automatically when called on the base class.

    Parameters
    ----------
    data : `~numpy.ndarray` or `FTSData`
        The actual data contained in this `FTSData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.

    uncertainty : `~astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.data.wcs.world_axis_physical_types


    mask : `~numpy.ndarray`-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.

    hits : `~numpy.ndarray`-like, optional
        Hit map for the data, given as a int Numpy array or any object that
        can be converted to a int Numpy array with a shape
        matching that of the data.

    flags : `~numpy.ndarray`-like or `~astropy.nddata.FlagCollection`, optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type (or an object which can be converted
        to a Numpy array) with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.

    wcs : `~astropy.wcs.WCS`, optional
        WCS-object containing the world coordinate system for the data.

    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.

    unit : `~astropy.units.UnitBase` instance or str, optional
        The units of the data.

    """

    def opd_trim(self, fraction=0.99):
        axis = self._non_opd_axis_numpy_index
        mostly_good_opd = (~np.isnan(self.data) & ~self.mask).sum(axis)
        mostly_good_opd = np.abs(mostly_good_opd / np.median(mostly_good_opd[mostly_good_opd != 0]) - 1) < fraction

        _slice = slice(*np.nonzero(mostly_good_opd)[0][[0, -1]])

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[self._opd_axis_index] -= _slice.start

        meta = deepcopy(self.meta)
        meta["history"] = f"opd_trim {fraction}"

        if self._opd_axis_numpy_index != 0:
            _slice = tuple([slice(None)] * self._opd_axis_numpy_index + [_slice])

        data = self.data[_slice]
        mask = self.mask[_slice] if self.mask is not None else None
        hits = self.hits[_slice] if self.hits is not None else None
        uncertainty = VarianceUncertainty(self.weights[_slice]) if self._uncertainty is not None else None

        result = self.__class__(data, wcs=wcs, mask=mask, meta=meta, hits=hits, uncertainty=uncertainty)

        return result

    # from CCDData
    def _slice_wcs(self, item):
        """
        Override the WCS slicing behaviour so that the wcs attribute continues
        to be an `astropy.wcs.WCS`.
        """
        if self.wcs is None:
            return None

        try:
            return self.wcs[item]
        except Exception as err:
            self._handle_wcs_slicing_error(err, item)

    def _extract_doublesided(self):
        """Return the largest doublesided OPD cube from the data.

        Returns
        -------
        output : DoubleSidedOpdData
            A doublesided interferograms cube
        """

        assert self._has_wcs_opd, "Intput should be OPD cube"

        opds = self.opd_axis.value

        _maxopd = np.min([-opds.min(), opds.max()])

        opd_wcs = self.opd_wcs
        signed = np.sign(opd_wcs.wcs.cdelt[0])
        slice_idx = np.round(opd_wcs.all_world2pix([-signed * _maxopd, signed * _maxopd], 0)[0]).astype(int)
        slice_idx += [0, 1]  # Inclusive end
        _slice = slice(*slice_idx)

        wcs = deepcopy(self.wcs)
        wcs.wcs.crpix[self._opd_axis_index] -= _slice.start

        meta = deepcopy(self.meta)
        meta["history"] = "extract_doublesided"

        if self._opd_axis_numpy_index != 0:
            _slice = tuple([slice(None)] * self._opd_axis_numpy_index + [_slice])

        data = self.data[_slice]
        mask = self.mask[_slice] if self.mask is not None else None
        hits = self.hits[_slice] if self.hits is not None else None
        uncertainty = VarianceUncertainty(self.weights[_slice]) if self._uncertainty is not None else None

        result = DoubleSidedOpdData(data, wcs=wcs, mask=mask, meta=meta, hits=hits, uncertainty=uncertainty)
        return result

    def _to_onesided(self, fold=True, long=True, alpha=0, **kwargs):
        """Return a onesided OPD cube from the data.

        Output will have long OPD on the right hand side.

        Parameters
        ----------
        fold : bool
            Fold/average the other side back, default=True
        long : bool
            Return the longest side, default=True
        alpha : float
            Apodization factor, default=0.1, see Notes

        Returns
        -------
        output : OneSidedOpdData
            A onesided interferograms cube

        Notes
        ------
        The apodization function used is the Tukey window, see [1]. Using `alpha=0.1` is downweighting 5% of the edges of the interferograms.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        extract_slice, os_slice, db_slice = self._to_onesided_slices(long=long)

        ma_data = np.ma.array(self.data.copy(), mask=self.mask)
        if self._uncertainty is not None:
            ma_weights = np.ma.array(self.weights, mask=self.mask)
        else:
            ma_weights = np.ma.array(~self.mask, mask=self.mask, dtype=np.float32)

        # Extract the longest part
        extract_itg = ma_data[extract_slice].filled(0).copy()
        extract_weight = ma_weights[extract_slice].filled(0)
        opd_idx = self._opd_axis_numpy_index
        naxis = extract_weight.shape[opd_idx]
        # downweight the edges, avoid weight 0 at the last point
        extract_apodization = windows.tukey(2 * naxis + 1, alpha=alpha)[naxis:-1]
        expand_axis = tuple([idx for idx in range(len(extract_weight.shape)) if idx != opd_idx])
        extract_weight *= np.expand_dims(extract_apodization, axis=expand_axis)

        # Weighted average :
        extract_itg *= extract_weight

        if fold:
            # Take the mean with the other half on the double sided part, taking the mask/weight into account
            side_itg = ma_data[db_slice].filled(0)
            side_weight = ma_weights[db_slice].filled(0)

            # Apodization of the side
            naxis = side_weight.shape[opd_idx]
            # downweight the edges, avoid weight 0 at the last point
            side_apodization = windows.tukey(2 * naxis + 1, alpha=alpha)[naxis:-1]
            side_weight *= np.expand_dims(side_apodization, axis=expand_axis)

            side_itg *= side_weight

            extract_itg[os_slice] += side_itg
            extract_weight[os_slice] += side_weight

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            extract_itg /= extract_weight

        uncertainty = InverseVariance(extract_weight)

        extract_hits = None
        if self._hits is not None:
            ma_hits = np.ma.array(self._hits.copy(), mask=self.mask)
            extract_hits = ma_hits[extract_slice].filled(0)
            if fold:
                side_hits = ma_hits[db_slice].filled(0)
                extract_hits[os_slice] += side_hits

        extract_wcs = deepcopy(self.wcs)
        extract_wcs.wcs.crpix[self._opd_axis_index] = 1

        meta = self.meta.copy()
        meta["fold"] = fold
        meta["long"] = long
        meta["alpha"] = alpha

        output = OneSidedOpdData(
            extract_itg,
            wcs=extract_wcs,
            mask=np.isnan(extract_itg),
            meta=meta,
            uncertainty=uncertainty,
            hits=extract_hits,
        )

        return output

    def _invert_uncertainty(self, output_shape=1):
        """Return the un-normed inverted uncertainty.

        Returns
        -------
        uncertainty : VarianceUncertainty
            the 2D uncertainty of the inverted cube
        output_shape : int
            the output shape in the frequency dimension

        Notes
        -----
        The uncertainty is computed as the sum of the variance of the input cube, along the fft axis
        """

        opd_idx = self._opd_axis_numpy_index
        uncertainty = self._uncertainty.represent_as(VarianceUncertainty).array

        # insure masked voxels are nan
        if np.any(self.mask):
            uncertainty[self.mask] = np.nan

        # Sum of variance along the opd axis
        # divided by naxis_opd**2 because we are dealing with ifft
        uncertainty = np.nansum(uncertainty, axis=opd_idx) / output_shape**2

        # Force empty pixels uncertainty to nan
        # uncertainty[uncertainty == 0] = np.nan

        return VarianceUncertainty(uncertainty)

    def _invert_hits(self):
        """Return the total number of hits used per pixels"""
        if self._hits is None:
            return None
        opd_idx = self._opd_axis_numpy_index
        hits = self._hits.copy()
        # insure masked voxels are nan
        hits[self.mask] = 0
        return np.nansum(hits, axis=opd_idx)

    def _make_extra_meta(self, output_shape, apodization, apodization_key):
        """Helper to create extra_meta dict for inversion methods."""
        opd_wcs = self.opd_wcs
        cdelt_opd = opd_wcs.wcs.cdelt[0]
        cunit_opd = u.Unit(opd_wcs.wcs.cunit[0])
        return dict(
            dopd=str(cdelt_opd * cunit_opd),
            output_shape=str(output_shape) if output_shape is not None else "None",
            opd_max=str(self._opd_max),
            **{
                apodization_key: (
                    f"{apodization.__module__}.{apodization.__name__}"
                    if callable(apodization)
                    else "None"
                )
            },
        )

    def _phase_error_corrected(
        self,
        niter=1,
        doublesided_apodization=None,
        medfilt_size=None,
        deg=1,
        fitting_func="polynomial",
        pcf_apodization=None,
        **kwargs,
    ):
        """Correct the phase error in the interferograms cube.

        Parameters
        ----------
        niter : [int], optional
            number of iterations, by default 1
        doublesided_apodization : [function], optional
            apodization function for the double sided inversion, by default None, but see Notes
        medfilt_size : [int], optional
            size of the median filtering window to be applied (before polynomial fitting), by default None
        deg : [int]
            the polynomial degree to fit to the phase, by default 1
        fitting_func : [str], ("polynomial"|"chebysev"), optional
            fitting function class, either polynomial or chebyshev, by default, "polynomial"
        pcf_apodization : [function], optional
            apodization function for the phase correction function, by default None

        Returns
        -------
        output : SingleSidedOpdData
            The corresponding spectral cube
        """
        get_kwargs = dict(
            niter=niter,
            apodization=doublesided_apodization,
            medfilt_size=medfilt_size,
            deg=deg,
            fitting_func=fitting_func,
            pcf_apodization=pcf_apodization,
        )
        get_kwargs.update(**kwargs)
        doublesided = self._extract_doublesided()
        phase_correction_function = doublesided._get_phase_correction_function(**get_kwargs)

        if phase_correction_function is None:
            return self

        # Convolved the interferograms and hits
        itg = np.ma.array(self.data, mask=self.mask).filled(0)
        if self._uncertainty is not None:
            ma_weights = np.ma.array(self.weights, mask=self.mask)
        else:
            ma_weights = np.ma.array(~self.mask, mask=self.mask, dtype=np.float32)
        itg *= ma_weights.filled(0)

        opd_idx = self._opd_axis_numpy_index

        corrected_itg = fftconvolve(itg, phase_correction_function, mode="same", axes=opd_idx).real
        corrected_weights = fftconvolve(ma_weights, phase_correction_function, mode="same", axes=opd_idx).real

        # Normalize the corrected interferograms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corrected_itg /= corrected_weights

        corrected_hits = None
        if self.hits is not None:
            hits = np.ma.array(self.hits, mask=self.mask).filled(0)
            corrected_hits = fftconvolve(hits, phase_correction_function, mode="same", axes=opd_idx).real

        meta = self.meta.copy()
        for key, value in get_kwargs.items():
            _insert_in_metadata_fits_safe(meta, key, value if value is not None else "None")

        itg = self.__class__(
            corrected_itg,
            wcs=self.wcs,
            meta=meta,
            hits=corrected_hits,
            uncertainty=InverseVariance(corrected_weights),
        )

        return itg

    @property
    def _opd_mean_mask(self):
        """Return a mean mask value along the opd axis"""
        # Retrieve the mean mask over the opd axis (averaging over all other axis)

        if self.mask is not None:
            fully_flagged = self.mask.all(axis=self._opd_axis_numpy_index)
            fully_flagged = np.broadcast_to(fully_flagged, self.mask.shape)
        else:
            fully_flagged = False

        opd_mean_mask = np.ma.array(self.mask, mask=fully_flagged).mean(axis=self._non_opd_axis_numpy_index)
    
        if np.ma.median(opd_mean_mask) > 0.1:
            self.__log.error("Most of the line of sight have 10% flagged data")
        elif np.ma.median(opd_mean_mask) > 0.05:
            self.__log.warning("Most of the line of sight have 5% flagged data")

        return opd_mean_mask

    def to_spectra(
        self, 
        apodization=None, 
        output_shape=None, 
        n_pad=None,
        peak=False,
        alpha=0,
        fold=True,
        long=True,
        phase_error_correction=False, 
        doublesided_apodization=None,
        **kwargs
    ):
        """Invert an interferograms cube using the (enhanced) Forman method.

        This generic implementation is kept for retro-compatibility (it is used
        by :class:`FTSData`).  The geometry-specific subclasses
        :class:`DoubleSidedOpdData`, :class:`OneSidedOpdData` and
        :class:`SingleSidedOpdData` override it with a more direct inversion.

        Parameters
        ----------
        apodization : [function], optional
            apodization function to be used on the one sided interferograms, by default None
        output_shape : int, optionnal
            the desired output shape for the spectrum axis (see Note)
        n_pad : int, optionnal
            the number of time to pad the input spectrum (for the final spectra only)
        peak : bool
            If True, the output will be normalized to the peak of the spectrum, default=False
        alpha : float
            Apodization factor used when converting to one-sided, default=0
        fold : bool
            Fold/average the other side back, default=True
        long : bool
            Use the longest side, default=True
        phase_error_correction : bool
            perform the phase error correction following the enhanced Forman method, by default False
        doublesided_apodization : [function], optional
            apodization function for the double sided inversion, by default None, but see Notes
        niter : [int], optional
            number of iterations, by default 1
        medfilt_size : [int], optional
            size of the median filtering window to be applied (before polynomial fitting), by default None
        deg : [int], optional
            the polynomial degree to fit to the phase, by default None
        pcf_apodization : [function], optional
            apodization function for the phase correction function, by default None

        Returns
        -------
        output : SpecData
            The corresponding spectral cube

        Notes
        -----
        Choice of apodization function can be made among the function available in numpy at [1]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        The output cube will be output_shape // 2 + 1, to retain only positive frequency

        References
        ----------
        .. [1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
               Photometer, 2005MsT..........1S
        .. [2] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        if phase_error_correction:
            logger.info(
                "Computing phase error correction (params=%s)",
                {k: kwargs[k] for k in ("niter", "medfilt_size", "deg", "fitting_func") if k in kwargs},
            )
            itg = self._phase_error_corrected(doublesided_apodization=doublesided_apodization, **kwargs)
        else:
            itg = self

        logger.info(
            "Converting interferograms to onesided (fold=%s, long=%s, alpha=%s)",
            fold,
            long,
            alpha,
        )
        onesided = itg._to_onesided(fold=fold, long=long, alpha=alpha)

        logger.info(
            "Onesided inversion : shape=%s, apodization=%s, output_shape=%s, n_pad=%s",
            self.shape,
            apodization.__name__ if callable(apodization) else apodization,
            output_shape,
            n_pad,
        )
        spectra = onesided.to_spectra(
            apodization=apodization,
            output_shape=output_shape,
            n_pad=n_pad,
            peak=peak,
        )

        return spectra

    @classmethod
    def from_array(cls, opd, data, hits=None, mask=None, **kwargs):
        """Construct FTS data from arrays.

        Parameters
        ----------
        opd : array_like or Quantity (M,)
            the optical path difference, by default 'mm'
        data : array_like (M, ...)
            the corresponding data, first dimension must match opd
        hits : array_like, optionnal
            the corresponding hits
        mask : array_like, optionnal
            the corresponding mask

        Returns
        -------
        data : SingleSidedOpdData
            the corresponding OpdData objects.  When called on the base
            :class:`SingleSidedOpdData` class, the returned object is an instance of
            :class:`DoubleSidedOpdData`, :class:`OneSidedOpdData` or
            :class:`SingleSidedOpdData` depending on the OPD geometry.
        """

        naxis = len(data.shape)
        wcs = WCS(naxis=naxis)

        if not isinstance(opd, u.Quantity):
            opd = u.Quantity(opd, "mm")

        zpd_idx = np.argmin(np.abs(opd))

        if opd[zpd_idx] != 0:
            cls.__log.info("Shifting opd by {} for 0".format(opd[zpd_idx]))
            opd -= opd[zpd_idx]

        dpd = np.diff(opd)
        np.testing.assert_almost_equal(
            np.median(dpd).to(dpd.unit).value, dpd.value, err_msg="Problem on opd differences"
        )

        wcs.wcs.ctype[naxis - 1] = "OPD"
        wcs.wcs.cunit[naxis - 1] = opd.unit

        wcs.wcs.crpix[naxis - 1] = zpd_idx + 1
        wcs.wcs.crval[naxis - 1] = opd[zpd_idx].value
        wcs.wcs.cdelt[naxis - 1] = np.median(dpd).value

        if cls is SingleSidedOpdData:
            cls = _classify_opd(wcs, shape=data.shape)

        if mask is None:
            mask = False

        return cls(data, wcs=wcs, hits=hits, mask=mask | np.isnan(data), **kwargs)

@logged
class DoubleSidedOpdData(SingleSidedOpdData):
    """Interferogram cube with a symmetric OPD axis, ZPD at the centre.

    This is the usual case for a full double-sided FTS scan.  The spectrum is
    obtained by a direct (centred) FFT of the whole interferogram, see
    :meth:`to_spectra`.
    """

    def _apodization(self, apodization=None, opd_max=None):
        """Return the full-window apodization array.

        Parameters
        ----------
        apodization : callable, optional
            Window function following the ``numpy`` convention (taking the
            number of points as argument), by default a boxcar.
        opd_max : `~astropy.units.Quantity`, optional
            Ignored for a double-sided interferogram (the full window is used).

        Returns
        -------
        apodization_array : `~numpy.ndarray`
            The apodization array along the OPD axis.
        """
        naxis_opd = self.naxis_opd

        if not callable(apodization):
            apodization = np.ones

        return apodization(naxis_opd)

    def to_spectra(
        self,
        apodization=None,
        output_shape=None,
        n_pad=None,
        peak=False,
        small_cube_bytes=1e7,
        ncpu=None,
        mem_available=None,
        use_dask=False,
        real=True,
        **kwargs,
    ):
        """Invert a double-sided interferogram cube with a direct FFT.

        Parameters
        ----------
        apodization : callable, optional
            Apodization function to be used on the full interferograms, by default None (boxcar)
        output_shape : int, optional
            the desired output shape for the spectrum axis (see Note)
        n_pad : int, optional
            the number of time to pad the input spectrum (see Note)
        peak : bool
            If True, the output will be normalized to the peak of the spectrum, default=False
        real : bool
            If True (default), return the real part of the spectrum.  Set to
            False to keep the complex spectrum (used by the phase-correction
            machinery).
        small_cube_bytes : float
            If the cube is smaller than this number of bytes, it will be processed in one go, otherwise in parallel blocks
        ncpu : int, optional
            Number of worker threads. Default is ``None``, which uses
            :func:`ftsdata._utils.cpu_count`.
        mem_available : float, optional
            Available memory, in bytes. Default is ``None``, which uses
            :func:`ftsdata._utils.available_memory`.
        use_dask : bool, optional
            If True, use dask to process the cube in parallel.
            If None, use dask if available.
            If False (default), use ThreadPoolExecutor.

        Returns
        -------
        output : SpecData
            The corresponding spectral cube

        Notes
        -----
        Choice can be made among the function available in numpy at [1]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        By default the output_shape is set to the number of points in the OPD axis.
        The output cube will be of shape output_shape // 2 + 1, to retain only positive frequency.
        Alternatively, one can use n_pad to increase the number of points in the output cube.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        assert self._has_wcs_opd, "Input should be OPD cube"
        assert self._is_doublesided, "Not a doublesided interferogram cube"

        opd_wcs = self.opd_wcs
        cdelt_opd = opd_wcs.wcs.cdelt[0]
        cunit_opd = u.Unit(opd_wcs.wcs.cunit[0])
        naxis_opd = self.naxis_opd

        if output_shape is None:
            output_shape = naxis_opd

        if n_pad is not None:
            output_shape = n_pad * (naxis_opd - 1) + 1 if naxis_opd % 2 == 1 else n_pad * naxis_opd + 1

        # Record the *resolved* output_shape (not None) in the metadata.
        meta = self.meta.copy()
        extra_meta = self._make_extra_meta(output_shape, apodization, "doublesided_apodization")
        for key, value in extra_meta.items():
            _insert_in_metadata_fits_safe(meta, key, value)

        n_out = output_shape // 2 + 1

        # Spencer 2005 Eq 2.29, direct fft
        # Trick is to use the unnormalized ifft to undo numpy's 1/n
        norm_factor = output_shape * (4 * cdelt_opd * cunit_opd / cst.c)
        norm_factor = norm_factor.to_value(1 / u.GHz)  # [Hz (electronic)/ GHz (electromagnetic)]
        if peak:
            norm_factor /= naxis_opd

        data = self.data
        real_dtype = np.float32 if data.dtype == np.float32 else np.float64
        # Keep the complex spectrum only when explicitly requested.
        out_dtype = real_dtype if (peak or real) else np.result_type(real_dtype, np.complex64)
        opd_idx = self._opd_axis_numpy_index

        apodization_array = self._apodization(apodization)
        weights = (np.asarray(apodization_array) * norm_factor).astype(real_dtype)

        # Per pixel: weighted copy + ifftshift copy + padded copy + complex ifft output + sliced copy
        bytes_work_per_pix = (2 * naxis_opd + 3 * output_shape + 2 * n_out) * data.dtype.itemsize

        block_func = partial(
            _to_spectra_doublesided_block,
            axis=opd_idx,
            output_shape=output_shape,
            weights=weights,
            peak=peak,
            real=real,
        )

        spectra = apply_blockwise(
            block_func,
            data,
            opd_idx,
            n_out,  # n_out = output_shape // 2 + 1
            mask=self.mask,
            out_dtype=out_dtype,
            bytes_work_per_pix=bytes_work_per_pix,
            small_cube_bytes=small_cube_bytes,
            ncpu=ncpu,
            mem_available=mem_available,
            use_dask=use_dask,
        )

        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = self._invert_uncertainty(output_shape=output_shape).represent_as(StdDevUncertainty).array
            # Normalisation of the uncertainty
            unc_factor = norm_factor  # unit change
            if peak or real:
                # We only keep the real part of the spectrum: the noise is split
                # between the real and imaginary parts.
                unc_factor = unc_factor / np.sqrt(2)
            uncertainty *= unc_factor
            uncertainty = StdDevUncertainty(uncertainty)

        hits = self._invert_hits()

        # Build new spectral wcs
        wcs = deepcopy(self.wcs)
        wcs_opd_idx = self._opd_axis_index
        wcs.wcs.ctype[wcs_opd_idx] = "FREQ"
        wcs.wcs.cunit[wcs_opd_idx] = "Hz"
        wcs.wcs.cdelt[wcs_opd_idx] = (cst.c / (cdelt_opd * cunit_opd) / output_shape).to_value(u.Hz)
        wcs.wcs.crpix[wcs_opd_idx] = 1
        wcs.wcs.crval[wcs_opd_idx] = 0

        unit = self.unit
        if unit is not None:
            unit = unit / u.GHz

        if "UNIT" in meta:
            meta["UNIT"] += "/GHz"

        # TODO: Add the mask !!
        output = SpecData(spectra, meta=meta, wcs=wcs, unit=unit, uncertainty=uncertainty, hits=hits)
        return output

    def _get_phase_correction_function(
        self,
        niter=1,
        apodization=None,
        medfilt_size=None,
        deg=1,
        fitting_func="polynomial",
        pcf_apodization=None,
        plot=False,
        **kwargs,
    ):
        """Compute the phase correction function for the current cube

        This follow the description in [1]_ with some additionnal features.

        Parameters
        ----------
        niter : [int], optional
            number of iterations, by default 1
        apodization : [function], optional
            apodization function for the double sided inversion, by default None, but see Notes
        medfilt_size : [int], optional
            size of the median filtering window to be applied (before polynomial fitting), by default None
        deg : [int]
            the polynomial degree to fit to the phase, by default 1
        fitting_func : [str], ("polynomial"|"chebysev"), optional
            fitting function class, either polynomial or chebyshev, by default, "polynomial"
        pcf_apodization : [function], optional
            apodization function for the phase correction function, by default None
        plot : bool, optional
            diagnostic plots, by default False

        Returns
        -------
        array_like (cube shape)
            the phase correction function to be used as convolution kernel for the interferograms

        Notes
        -----
        Choice of apodization function can be made among the function available in numpy at [2]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.


        References
        ----------
        .. [1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
               Photometer, 2005MsT..........1S
        """
        if pcf_apodization is None:
            pcf_apodization = np.ones

        # Working copy
        itg = deepcopy(self)

        # Reference iterferogram
        itg_ma = np.ma.array(itg.data, mask=itg.mask, copy=True).filled(0)

        # Null starting phase (take only the upper part)
        # upper_slice = slice((itg.shape[0] - 1) // 2, None)
        # phase = np.zeros((slice_length(upper_slice, itg.shape[0]), *itg.shape[1:]))
        phase_shape = list(itg.shape)
        freq_shape = itg.shape[self._opd_axis_numpy_index] // 2 + 1
        phase_shape[self._opd_axis_numpy_index] = freq_shape
        phase = np.zeros(shape=phase_shape)

        pcf = None

        output_shape = 2 * (freq_shape - 1) + 1
        pcf_apodization = pcf_apodization(output_shape)

        # Loop Here
        for i in range(niter):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=VerifyWarning)
                cube = itg.to_spectra(
                    apodization=apodization, output_shape=output_shape, peak=False, real=False
                )

            # Spencer 2.39 , well actually phases are -pi/pi so arctan2 or angle
            # TODO: This is where the amplitude get flipped if we have a negative spectrum (abs will be >0 and phase < 0)
            _phase = np.unwrap(np.angle(cube.data), axis=self._opd_axis_numpy_index)

            # Replace bad phase :
            _phase[np.isnan(_phase)] = 0

            if plot:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(ncols=4)
                freq = cube.spectral_axis.value
                axes[1].plot(freq, cube.data[:, :, 0])
                axes[2].plot(freq, _phase[:, :, 0])

            if medfilt_size is not None:
                # Median filtering of the phases
                kernel_size = np.ones(len(itg.shape), dtype=int)
                kernel_size[self._opd_axis_numpy_index] = medfilt_size
                _phase = medfilt(_phase, kernel_size=kernel_size)

            if deg is not None:
                if fitting_func == "polynomial":
                    polyfit, polyval = np.polynomial.polynomial.polyfit, np.polynomial.polynomial.polyval
                elif fitting_func == "chebychev":
                    polyfit, polyval = np.polynomial.chebyshev.chebfit, np.polynomial.chebyshev.chebval
                else:
                    raise ValueError('fitting_func should be in ("polynomial"|"chebychev")')

                # polynomial fit on the phase, weighted by the intensity
                p = []
                idx = np.linspace(0, 1, freq_shape)
                # swap axes to get frequency on first axis
                _phase = _phase.swapaxes(0, self._opd_axis_numpy_index)
                _weight = np.abs(cube.data).swapaxes(0, self._opd_axis_numpy_index)
                # np.polynomial.polynomial.polyfit do not accept a (`M`, `K`) array for the weights, so need to loop....
                for spec, weight in zip(
                    _phase.reshape(freq_shape, -1).T,
                    _weight.reshape(freq_shape, -1).T,
                ):
                    # Weights are not enough to discard the edges, need to cut
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        good = weight / np.max(weight) > 0

                    if not np.any(good):
                        good = np.ones_like(idx, dtype=bool)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RankWarning)
                        coef = polyfit(idx[good], spec[good], deg, w=weight[good] ** 2)

                    # Null the offset to avoid sign flip
                    coef[0] = 0
                    p.append(coef)

                p = np.asarray(p).T

                # evaluate the polynomal all at once :
                _phase = polyval(idx, p).T.reshape(_phase.shape).swapaxes(0, self._opd_axis_numpy_index)

                # Wrap back the phases to -pi pi
                _phase = (_phase + np.pi) % (2 * np.pi) - np.pi
                # Actually phase must be within -pi/2 pi/2 otherwise a flip can occur, this fails in tests
                # _phase = (_phase + np.pi / 2) % np.pi - np.pi / 2
                """
fit data also incorporates smoothing in the
out of band region to ensure zero phase and derivative discontinuities and zero amplitude at
zero and Nyquist frequency.
                """

            if plot:
                axes[2].plot(freq, _phase[:, :, 0], linestyle="--")

            phase += _phase

            # Spencer 3.30
            # pcf = np.fft.fftshift(np.fft.irfft(np.exp(1j * phase), axis=0, n=2 * (phase.shape[0] - 1) + 1), axes=0)
            pcf = np.fft.fftshift(
                np.fft.irfft(np.exp(1j * phase), axis=self._opd_axis_numpy_index, n=output_shape),
                axes=self._opd_axis_numpy_index,
            )
            # full_phase = np.concatenate([-phase[:0:-1], phase])
            # pcf = np.fft.fftshift(
            #     np.fft.ifft(np.exp(1j * np.fft.ifftshift(full_phase, axes=0)), axis=0),
            #     axes=0,
            # ).real

            # Apodization of the PCF along the freq axis
            # pcf = (pcf.T * pcf_apodization(pcf.shape[self._opd_axis_numpy_index])).T
            pcf = np.apply_along_axis(lambda _: _ * pcf_apodization, self._opd_axis_numpy_index, pcf)

            if plot:
                (x,) = itg.wcs.sub([3]).all_pix2world(np.arange(itg.shape[0]), 0)
                axes[3].plot(x, pcf[:, :, 0])
                axes[3].set_xlim(-1, 1)
                axes[0].plot(x, itg.data[:, :, 0])
                axes[0].set_xlim(-1, 1)

            # Correct the initial dataset with the current phase for the next iteration
            corrected_itg = fftconvolve(itg_ma, pcf, mode="same", axes=self._opd_axis_numpy_index).real
            itg.data[:] = corrected_itg

        return pcf

@logged
class OneSidedOpdData(SingleSidedOpdData):
    """Interferogram cube with the ZPD at the start of the OPD axis.

    Only positive OPDs are stored.  The spectrum is obtained by an inverse real
    FFT, see :meth:`to_spectra`.
    """

    def _apodization(self, apodization=None, opd_max=None):
        """Return the half-window apodization array.

        The window is only applied up to ``opd_max``; beyond that the
        interferogram is zeroed.

        Parameters
        ----------
        apodization : callable, optional
            Window function following the ``numpy`` convention (taking the
            number of points as argument), by default a boxcar.
        opd_max : `~astropy.units.Quantity`, optional
            Maximum OPD up to which the window is defined.  Defaults to
            :attr:`OpdAxisMixinClass._opd_max`.

        Returns
        -------
        apodization_array : `~numpy.ndarray`
            The apodization array along the OPD axis.
        """
        naxis_opd = self.naxis_opd

        if not callable(apodization):
            apodization = np.ones

        if opd_max is None:
            opd_max = self._opd_max

        naxis_apod = self.opd_wcs.world_to_array_index(opd_max)

        apodization_array = np.zeros(naxis_opd)
        apodization_array[:naxis_apod] = apodization(2 * naxis_apod - 1)[naxis_apod - 1 :]

        return apodization_array

    def to_spectra(
        self,
        apodization=None,
        output_shape=None,
        n_pad=None,
        peak=False,
        small_cube_bytes=1e7,
        ncpu=None,
        mem_available=None,
        use_dask=False,
        **kwargs,
    ):
        """Invert a one-sided interferogram cube.

        Parameters
        ----------
        apodization : callable, optional
            Apodization function to be used on the one-sided interferograms, by default None
        output_shape : int, optional
            the desired output shape for the spectrum axis (see Note)
        n_pad : int, optional
            the number of time to pad the input spectrum (see Note)
        peak : bool
            If True, the output will be normalized to the peak of the spectrum, default=False
        small_cube_bytes : float
            If the cube is smaller than this number of bytes, it will be processed in one go, otherwise in parallel blocks
        ncpu : int, optional
            Number of worker threads. Default is ``None``, which uses
            :func:`ftsdata._utils.cpu_count`.
        mem_available : float, optional
            Available memory, in bytes. Default is ``None``, which uses
            :func:`ftsdata._utils.available_memory`.
        use_dask : bool, optional
            If True, use dask to process the cube in parallel.
            If None (default), use dask if available.
            If False, use ThreadPoolExecutor.

        Returns
        -------
        output : SpecData
            The corresponding spectral cube

        Notes
        -----
        Choice can be made among the function available in numpy at [1]_, namely
        `numpy.hanning`, `numpy.hamming`, `numpy.bartlett`, `numpy.blackman`, `numpy.kaiser`
        or any custom routine following the same convention.

        By default the output_shape is set to 2 * (naxis_opd - 1).
        The output cube will be of shape output_shape // 2 + 1, to retain only positive frequency.
        Alternatively, one can use n_pad to increase the number of points in the output cube.

        References
        ----------
        .. [1] https://docs.scipy.org/doc/numpy/reference/routines.window.html
        """
        assert self._has_wcs_opd, "Input should be OPD cube"
        assert self._is_onesided, "Not a one sided interferogram cube"

        opd_wcs = self.opd_wcs
        cdelt_opd = opd_wcs.wcs.cdelt[0]
        cunit_opd = u.Unit(opd_wcs.wcs.cunit[0])
        naxis_opd = self.naxis_opd

        if output_shape is None:
            # Add +1 to match the doublesided case where naxis_opd is odd to allow for central zpd pixel
            output_shape = 2 * (naxis_opd - 1) + 1

        if n_pad is not None:
            output_shape = n_pad * 2 * (naxis_opd - 1) + 1

        # Record the *resolved* output_shape (not None) in the metadata.
        meta = self.meta.copy()
        extra_meta = self._make_extra_meta(output_shape, apodization, "onesided_apodization")
        for key, value in extra_meta.items():
            _insert_in_metadata_fits_safe(meta, key, value)

        n_out = output_shape // 2 + 1

        # Spencer 2005 Eq 2.29, direct fft
        # Trick is to use the unnormalized irfft to undo numpy's 1/n
        norm_factor = output_shape * (4 * cdelt_opd * cunit_opd / cst.c)
        norm_factor = norm_factor.to_value(1 / u.GHz)  # [Hz (electronic)/ GHz (electromagnetic)]
        if peak:
            norm_factor /= 2 * (naxis_opd - 1) + 1

        data = self.data
        out_dtype = np.float32 if data.dtype == np.float32 else np.float64
        opd_idx = self._opd_axis_numpy_index

        apodization_array = self._apodization(apodization)
        weights = (np.asarray(apodization_array) * norm_factor).astype(out_dtype)

        # Per pixel: masked/weighted copy + complex input (padded) + irfft output + sliced copy
        bytes_work_per_pix = (naxis_opd + 3 * n_out + output_shape) * data.dtype.itemsize

        block_func = partial(_to_spectra_onesided_block, axis=opd_idx, output_shape=output_shape, weights=weights)

        spectra = apply_blockwise(
            block_func,
            data,
            opd_idx,
            n_out,  # n_out = output_shape // 2 + 1
            mask=self.mask,
            out_dtype=out_dtype,
            bytes_work_per_pix=bytes_work_per_pix,
            small_cube_bytes=small_cube_bytes,
            ncpu=ncpu,
            mem_available=mem_available,
            use_dask=use_dask,
        )

        uncertainty = None
        if self.uncertainty is not None:
            uncertainty = self._invert_uncertainty(output_shape=output_shape).represent_as(StdDevUncertainty).array
            unc_factor = norm_factor * np.sqrt(2)  # Counted twice
            uncertainty *= unc_factor
            uncertainty = StdDevUncertainty(uncertainty)

        hits = self._invert_hits()

        # Build new wcs
        wcs = deepcopy(self.wcs)
        wcs_opd_idx = self._opd_axis_index
        wcs.wcs.ctype[wcs_opd_idx] = "FREQ"
        wcs.wcs.cunit[wcs_opd_idx] = "Hz"
        wcs.wcs.cdelt[wcs_opd_idx] = (cst.c / (cdelt_opd * cunit_opd) / output_shape).to_value(u.Hz)
        wcs.wcs.crpix[wcs_opd_idx] = 1
        wcs.wcs.crval[wcs_opd_idx] = 0

        unit = self.unit
        if unit is not None:
            unit = unit / u.GHz

        if "UNIT" in meta:
            meta["UNIT"] += "/GHz"

        # TODO: Add the mask !!
        output = SpecData(spectra, meta=meta, wcs=wcs, unit=unit, uncertainty=uncertainty, hits=hits)

        return output

def _classify_opd(wcs, shape=None):
    """Return the :class:`OpdData` subclass matching an OPD WCS geometry.

    The classification follows the same rules as the
    :attr:`OpdAxisMixinClass._is_doublesided` and
    :attr:`OpdAxisMixinClass._is_onesided` properties:

    - the OPD axis is double-sided when the OPD values sum to zero (ZPD centred);
    - it is one-sided when the first OPD value is zero (ZPD at the start);
    - otherwise it is single-sided (asymmetric).

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        WCS of the interferogram cube.
    shape : tuple, optional
        Shape of the data array.  Used when the WCS does not carry an
        ``array_shape`` (e.g. a freshly constructed WCS).

    Returns
    -------
    cls : type
        One of :class:`DoubleSidedOpdData`, :class:`OneSidedOpdData`,
        :class:`SingleSidedOpdData`, or :class:`OpdData` when the WCS has no
        OPD axis.
    """
    physical_type = wcs.world_axis_physical_types
    if "pos.opd" not in physical_type:
        return SingleSidedOpdData

    opd_axis_index = physical_type.index("pos.opd")
    opd_wcs = wcs.sub([opd_axis_index + 1])

    if shape is not None:
        naxis_opd = shape[-(opd_axis_index + 1)]
    elif wcs.pixel_shape is not None:
        naxis_opd = wcs.pixel_shape[opd_axis_index]
    elif wcs.array_shape is not None:
        naxis_opd = wcs.array_shape[-(opd_axis_index + 1)]
    else:
        naxis_opd = None

    cdelt_opd = opd_wcs.wcs.cdelt[0]

    if cdelt_opd <= 0 or naxis_opd is None or naxis_opd < 1:
        return SingleSidedOpdData

    opd_axis = opd_wcs.pixel_to_world(np.arange(naxis_opd)).value

    if np.isclose(opd_axis.sum(), 0):
        return DoubleSidedOpdData
    if opd_axis[0] == 0:
        return OneSidedOpdData
    return SingleSidedOpdData

@logged
class FTSData(SingleSidedOpdData,SpecData):
    """Retro-compatibility class.

    Use either SingleSidedOpdData, OneSidedOpdData, DoubleSidedOpdData or SpecData

    """

    pass
