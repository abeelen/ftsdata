"""Mixin classes shared by the ftsdata containers."""

import json
import logging
import warnings
from copy import deepcopy
from functools import cached_property, partial
from itertools import repeat
from typing import Callable, Optional

import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import InverseVariance
from astropy.nddata.ccddata import (
    _known_uncertainties,
    _unc_cls_to_name,
    _uncertainty_unit_equivalent_to_parent,
)
from astropy.wcs import WCS
from scipy.ndimage import binary_closing

from ._utils import (
    FITS_HEADER_COMMENT,
    _import_from,
    _insert_in_metadata_fits_safe,
)
from .transforms import (
    _to_itg_block,
    apply_blockwise,
    get_ils_kernel,
    get_itg_binning_correction,
)

logger = logging.getLogger("FTSData")

class OpdAxisMixinClass(object):
    """Mixin class to handle OPD coordinate in WCS."""

    @cached_property
    def _opd_axis_index(self):
        physical_type = self.wcs.world_axis_physical_types

        if "pos.opd" in physical_type:
            return physical_type.index("pos.opd")  # 0-indexed within the wcs properties
        else:
            return None

    @cached_property
    def _has_wcs_opd(self):
        return self._opd_axis_index is not None

    @cached_property
    def _opd_axis_numpy_index(self):
        if self._has_wcs_opd:
            return self.ndim - (self._opd_axis_index + 1)
        else:
            return None

    @cached_property
    def _non_opd_axis_numpy_index(self):
        return tuple([i for i in range(len(self.shape)) if i != self._opd_axis_numpy_index])

    @cached_property
    def opd_wcs(self):
        if self._has_wcs_opd:
            return self.wcs.sub([self._opd_axis_index + 1])
        else:
            raise ValueError("This object has no opd axis")

    @cached_property
    def naxis_opd(self):
        if self._has_wcs_opd:
            return self.shape[self._opd_axis_numpy_index]
        else:
            return None

    @cached_property
    def opd_axis(self):
        if self._has_wcs_opd:
            return self.opd_wcs.pixel_to_world(np.arange(self.naxis_opd))
        else:
            raise ValueError("This object has no opd axis")

    @property
    def _opd_max(self):
        """Determine the opd_max
        Use the full range of the cube
        """
        opd_mean_mask = self._opd_mean_mask
        fully_flagged = binary_closing(opd_mean_mask == 1, iterations=10, border_value=True)
        idxs = np.flatnonzero(~fully_flagged)
        idx_range = [idxs.min(), idxs.max()]
        # idx_range = [0, self.naxis_opd]
        opd_range = self.opd_wcs.pixel_to_world(idx_range)
        opd_max = np.max(np.abs(opd_range))
        return opd_max

    @property
    def _is_doublesided(self):
        """Return True is the cube is double sided, also enforce positive increments."""
        is_doublesided = np.isclose(self.opd_axis.value.sum(), 0)
        is_positive = self.opd_wcs.wcs.cdelt[0] > 0
        return is_doublesided & is_positive

    @property
    def _is_onesided(self):
        """Return True is the cube is one sided, also enforce positive increments."""
        is_onesided = self.opd_axis.value[0] == 0
        is_positive = self.opd_wcs.wcs.cdelt[0] > 0
        return is_onesided & is_positive

    def _to_onesided_slices(self, long=True):
        """Return the slices to extract onesided itg from the data."""

        if self._is_onesided:
            return self

        zpd_idx = int(self.opd_wcs.all_world2pix(0, 0)[0])

        extrema_opd = np.abs(self.opd_axis.value[[0, -1]])

        if extrema_opd[0] <= extrema_opd[1]:
            # Positive single sided : longer right hand side...
            # <--0----->
            # Or doublesided
            # <--0-->
            if long:
                # itg, zpd_idx = [-2, -1, 0, 1, 2, 3], 2
                # print(itg[extract_slice], itg[extract_slice][os_slice], itg[db_slice])
                # >> [0, 1, 2] [1] [-1]
                extract_slice = slice(zpd_idx, None)
                os_slice = slice(1, zpd_idx + 1)
                db_slice = slice(zpd_idx - 1, None, -1)
            else:
                # itg, zpd_idx = [-2, -1, 0, 1, 2, 3], 2
                # print(itg[extract_slice], itg[extract_slice][os_slice], itg[db_slice])
                # >> [0, -1, -2] [-1, -2] [1, 2]
                extract_slice = slice(zpd_idx, None, -1)
                os_slice = slice(1, None)
                db_slice = slice(zpd_idx + 1, 2 * zpd_idx + 1)
        elif extrema_opd[0] > extrema_opd[1]:
            naxis_opd = self.naxis_opd
            # Negative single sided : longer left hand side...
            # <-----0-->
            if long:
                # itg, zpd_idx = [-3, -2, -1, 0, 1, 2], 3
                # print(itg[extract_slice], itg[extract_slice][os_slice], itg[db_slice])
                # >> [0, -1, -2, -3] [0, -1, -2] [0, 1, 2]
                extract_slice = slice(zpd_idx, None, -1)
                os_slice = slice(1, naxis_opd - zpd_idx)
                db_slice = slice(zpd_idx + 1, None)
            else:
                # itg, zpd_idx = [-3, -2, -1, 0, 1, 2], 3
                # print(itg[extract_slice], itg[extract_slice][os_slice], itg[db_slice])
                # >> [0, 1, 2] [1, 2] [-1, -2]
                extract_slice = slice(zpd_idx, None)
                os_slice = slice(1, 2 * naxis_opd - zpd_idx)
                db_slice = slice(zpd_idx - 1, 2 * zpd_idx - naxis_opd, -1)

        if self._opd_axis_numpy_index == 0:
            return extract_slice, os_slice, db_slice
        else:
            # Slice on the proper axis
            return (
                tuple([slice(None)] * self._opd_axis_numpy_index + [item])
                for item in [extract_slice, os_slice, db_slice]
            )




class SpectralAxisMixinClass(object):
    """Mixin class to handle spectral coordinate in WCS."""

    @cached_property
    def _spectral_axis_index(self):
        physical_type = self.wcs.world_axis_physical_types

        if "em.freq" in physical_type:
            return physical_type.index("em.freq")  # 0-indexed within the wcs properties
        else:
            return None

    @cached_property
    def _has_wcs_spectral(self):
        return self._spectral_axis_index is not None

    @cached_property
    def _spectral_axis_numpy_index(self):
        if self._has_wcs_spectral:
            return self.ndim - (self._spectral_axis_index + 1)
        else:
            return None

    @cached_property
    def _non_spectral_axis_numpy_index(self):
        return tuple([i for i in range(len(self.shape)) if i != self._spectral_axis_numpy_index])

    @cached_property
    def spectral_wcs(self):
        if self._has_wcs_spectral:
            return self.wcs.sub([self._spectral_axis_index + 1])
        else:
            raise ValueError("This object has no spectral axis")

    @cached_property
    def spectral_axis(self):
        if self._has_wcs_spectral:
            naxis_spectral = self.shape[self._spectral_axis_numpy_index]
            return self.spectral_wcs.pixel_to_world(np.arange(naxis_spectral))
        else:
            raise ValueError("This object has no spectral axis")

    @cached_property
    def spectral_bins(self):
        if self._has_wcs_spectral:
            naxis_spectral = self.shape[self._spectral_axis_numpy_index]
            return self.spectral_wcs.pixel_to_world(np.arange(naxis_spectral + 1) - 0.5)
        else:
            raise ValueError("This object has no spectral axis")

    def _slice_item(self, item):
        """Return only the slices for the spatial dimension."""
        return tuple(_slice for idx, _slice in enumerate(item) if idx != self._spectral_axis_numpy_index)

    def _slice_uncertainty(self, item):
        if self.uncertainty is None:
            return None
        try:
            # Uncertainty are spatial 2D only...
            item = self._slice_item(item)
            return self.uncertainty[item]
        except (TypeError, KeyError):
            # Catching TypeError in case the object has no __getitem__ method.
            # Catching KeyError for Python 3.12.
            # But let IndexError raise.
            logger.info("uncertainty cannot be sliced.")
        return self.uncertainty

    def _slice_hits(self, item):
        if self.hits is None:
            return None
        try:
            item = self._slice_item(item)
            return self.hits[item]
        except (TypeError, KeyError):
            # Catching TypeError in case the object has no __getitem__ method.
            # Catching KeyError for Python 3.12.
            # But let IndexError raise.
            logger.info("hits cannot be sliced.")
        return self.hits

    def get_ils_kernel(self, dfreq_GHz=None):
        """Retrive the ILS from the meta data.

        Parameters
        ----------
        dfreq_GHz: float, optionnal
            the requested frequency scale in GHz, by default the one from the cube, see Notes

        Returns
        -------
        kernel : astropy.convolution.CustomKernel
            the kernel to be used for spectral convolution.

        Notes
        -----
        dfreq_GHz should be set to the spectrum resolution which will be used for convolution, before any resampling/averaging.
        """
        opdmax = self.meta.get("opd_max", None)
        assert opdmax is not None, "Could not find opd_max in meta data"
        opdmax = u.Quantity(opdmax)
        opdmax_mm = opdmax.to_value(u.mm)

        onesided_apodization = self.meta.get("onesided_apodization", None)
        if isinstance(onesided_apodization, str):
            onesided_apodization = _import_from(onesided_apodization)


        freq_GHz = self.spectral_axis.to_value("GHz")
        dfreq_GHz = np.median(np.diff(freq_GHz))

        return get_ils_kernel(opdmax_mm, dfreq_GHz, onesided_apodization=onesided_apodization)

    def _native_dopd(self):
        """Return the native dopd of the cube"""
        freq = self.spectral_axis
        dopd = cst.c / np.max(freq) / 2
        return dopd.to(u.mm)

    def _native_opdmax(self):
        """Return the native opdmax of the cube"""
        freq = self.spectral_axis
        dfreq = np.median(np.diff(freq))
        return (cst.c / (2 * dfreq)).to(u.mm)

    def to_itg(
        self,
        bandpass: Optional[Callable],
        center: Optional[SkyCoord] = None,
        dopd_mm=0.1,
        binning=True,
        small_cube_bytes=1e7,
        ncpu=None,
        mem_available=None,
        use_dask=None,
    ):
        """Convert the spectral cube into a cube of one-sided interferograms.

        Parameters
        ----------
        bandpass : callable or None
            Bandpass function to apply, optionnal
        center : astropy.coordinates.SkyCoord, optional
            New celestial center of the output cube
        dopd_mm : float, optional
            Requested OPD sampling step, in mm. Default is 0.1.
        binning : bool
            Do we need to apply a itg binning correction, default True
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
        OneSidedOpdData
            Interferogram cube.

        Raises
        ------
            ValueError
            If the cube extends above the Nyquist frequency c / (2 * dopd_mm)
            (negative end padding).

        Notes
        -----
        - The input cube must not contain NaNs
        - binning=False performs a simple sampling of the interferograms
        - dopd_mm will be adjusted to the actual value used in the output cube, which may differ from the requested value if the input cube is not sampled enough.
        """
        header = self.wcs.to_header()

        if center is not None:
            axis_types = WCS(header).world_axis_physical_types
            for axis_type, value in zip(["pos.eq.ra", "pos.eq.dec"], [center.ra.deg, center.dec.deg]):
                axis_index = axis_types.index(axis_type) + 1
                header[f"CRVAL{axis_index}"] = value

        freq = self.spectral_axis
        spec_npidx = self._spectral_axis_numpy_index
        nchan = len(freq)

        dfreq = np.median(np.diff(freq))

        # /2 Because we use only positive opds
        norm_factor = (dfreq / 2).to_value(u.GHz)

        # Cube needs to start at 0 GHz for the FFT
        pad_start = int(round((freq[0] / dfreq).decompose().value))  # to go from 0 to freq_min_GHz-dfreq_GHz

        # Depending on the requested resolution we may have to pad the cube at the end...
        freq_max = cst.c / (dopd_mm * u.mm) / 2
        nfreq_tot = int(np.floor((freq_max / dfreq).decompose())) + 1
        n = 2 * (nfreq_tot - 1)

        dopd_eff = (cst.c / (dfreq * n)).to(u.mm)  # actual step

        # One 1D weight vector: bandpass x binning sinc (no cube-sized temporaries)
        weights = np.ones_like(freq.data)
        if bandpass is not None:
            weights = weights * bandpass(freq.to_value("GHz"))
        if binning:
            weights = weights * get_itg_binning_correction(freq, dopd_eff)
        weights = weights.astype(np.float32)

        data = self.data
        if np.iscomplexobj(data):
            # An inverse FFT needs a real spectrum.  This happens when the
            # complex output of DoubleSidedOpdData.to_spectra(real=False) is
            # fed back into a one-sided interferogram.
            logger.warning("Discarding the imaginary part of the complex spectrum before to_itg")
            data = data.real
        out_dtype = np.float32 if data.dtype == np.float32 else np.float64
        bytes_work_per_pix = (nchan + 5 * nfreq_tot) * data.dtype.itemsize

        # Magic Shift for a cube that does not start at 0 GHz (phase in Fourier space).
        # The modulo keeps the angle small, so the phase stays accurate for large indices.
        # Computed once here and reused by every block; the /2 norm factor is folded in.
        # `itg` has the dtype scipy's rfft gives for this input, so build `phase_norm`
        # in that dtype and avoid a per-block cast in _to_itg_block.
        itg_dtype = np.result_type(data.dtype, np.complex64)
        m = np.arange(nfreq_tot)  # == n // 2 + 1
        phase_norm = np.exp(-2j * np.pi * ((pad_start * m) % n) / n)
        phase_norm *= norm_factor
        phase_norm = phase_norm.astype(itg_dtype)

        block_func = partial(
            _to_itg_block,
            axis=spec_npidx,
            n=n,
            phase_norm=phase_norm,
            weights=weights,
        )

        _itg = apply_blockwise(
            block_func,
            data,
            spec_npidx,
            nfreq_tot,
            out_dtype=out_dtype,
            bytes_work_per_pix=bytes_work_per_pix,
            small_cube_bytes=small_cube_bytes,
            ncpu=ncpu,
            mem_available=mem_available,
            use_dask=use_dask,
        )

        unit = self.unit
        if unit is not None:
            unit = unit * u.GHz

        # Build new WCS
        spec_idx = self._spectral_axis_index
        wcs = WCS(header)
        wcs.wcs.ctype[spec_idx] = "OPD"
        wcs.wcs.cunit[spec_idx] = "mm"
        wcs.wcs.cdelt[spec_idx] = dopd_eff.to_value(u.mm)
        wcs.wcs.crpix[spec_idx] = 1
        wcs.wcs.crval[spec_idx] = 0

        meta = self.meta.copy()
        if "UNIT" in meta:
            meta["UNIT"] = "GHz " + meta["UNIT"]

        meta["HISTORY"] = f"to_itg with dopd={dopd_mm} mm"
        # Imported here to keep the mixin module free of a core import cycle.
        from .core import OneSidedOpdData

        return OneSidedOpdData(data=_itg, wcs=wcs, meta=meta)


class WeightMixinClass(object):
    """Mixin class to define a weight property from the uncertainty"""

    @property
    def weights(self):
        """Return the weights as inverse variance regardless of the uncertainty type"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            uncert = self.uncertainty.represent_as(InverseVariance).array
        return uncert


class HitsMixinClass(object):
    """Mixin to add hits property"""

    _hits = None

    def __init__(self, *args, hits=None, **kwargs):
        # Initialize with the parent...
        super().__init__(*args, **kwargs)

        # Additionnal data
        self.hits = hits

    @property
    def hits(self):
        """
        ndarra : Hits in the dataset, if any.
        """
        return self._hits

    @hits.setter
    def hits(self, value):
        if value is not None:
            self._hits = np.asarray(value).astype(int)
        else:
            self._hits = None

    def _slice_hits(self, item):
        if self.hits is None:
            return None
        try:
            return self.hits[item]
        except (TypeError, KeyError):
            # Catching TypeError in case the object has no __getitem__ method.
            # Catching KeyError for Python 3.12.
            # But let IndexError raise.
            logger.info("hits cannot be sliced.")
        return self.hits

    def _slice(self, item):
        kwargs = super()._slice(item)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the sliced flags
        kwargs["hits"] = self._slice_hits(item)
        return kwargs  # these must be returned


class HDUMixinClass(object):
    """Add possibility to export cube to HDU FITS format"""

    def to_hdu(  # noqa: C901
        self,
        hdu_data="DATA",
        hdu_mask="MASK",
        hdu_uncertainty="UNCERT",
        hdu_flags=None,
        hdu_hits="HITS",
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
        compress=False,
    ):
        """Creates an HDUList object from a FTSData object.

        Parameters
        ----------
        hdu_data, hdu_mask, hdu_uncertainty, hdu_flags, hdu_hits : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Flags are not supported at this time. If ``None`` this attribute
            is not appended.
            Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty, ``'HITS'`` for hits and
            ``None`` for flags.

        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.

        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.

        compress : bool, optional
            Use tiled compression ImageHDU for the data. Default is ``False``.

        Raises
        ------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        NotImplementedError
            Saving flags is not supported.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
        if isinstance(self.meta, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.meta.copy()
        else:
            header = deepcopy(self.meta)
            history = header.pop("history", None)
            comment = header.pop("comment", None)

            # Based on nikamap.utils.meta_to_header
            dummy_header = fits.Header()
            for key, value in header.items():
                if key in ["history", "comment", "HISTORY", "COMMENT"]:
                    continue

                if isinstance(value, (u.Quantity)) or callable(value):
                    value = str(value)

                if not isinstance(
                    value, (int, float, str, complex, np.floating, np.integer, np.complexfloating, np.bool_)
                ):
                    value = json.dumps(value)

                _insert_in_metadata_fits_safe(dummy_header, key, value)

            for key in ["history", "comment"]:
                if key in header:
                    for item in header[key]:
                        dummy_header[key] = item

            header = dummy_header

            if history is not None:
                for item in history:
                    header["history"] = item
            if comment is not None:
                for item in comment:
                    header["comment"] = item

            for key, comment in FITS_HEADER_COMMENT.items():
                if key in header:
                    header.set(key, comment=comment)

        if self.unit is not None and self.unit is not u.dimensionless_unscaled:
            header["bunit"] = self.unit.to_string()

        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header(relax=wcs_relax)
            header.extend(wcs_header, useblanks=False, update=True)

        hdus = [fits.PrimaryHDU(header=header)]

        if compress:
            hduData = fits.CompImageHDU(self.data, header, name=hdu_data)
        else:
            hduData = fits.ImageHDU(self.data, header, name=hdu_data)
        hdus.append(hduData)

        if hdu_mask and self.mask is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.mask, "shape"):
                raise ValueError("only a numpy.ndarray mask can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            if compress:
                hduMask = fits.CompImageHDU(self.mask.astype(np.uint8), name=hdu_mask)
            else:
                hduMask = fits.ImageHDU(self.mask.astype(np.uint8), name=hdu_mask)
            hdus.append(hduMask)

        if hdu_uncertainty and self.uncertainty is not None:
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError("only uncertainties of type {} can be saved.".format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header()
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if hasattr(self.uncertainty, "unit") and self.uncertainty.unit is not None and self.unit is not None:
                if not _uncertainty_unit_equivalent_to_parent(uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        "saving uncertainties with a unit that is not "
                        "equivalent to the unit from the data unit is not "
                        "supported."
                    )

            if compress:
                hduUncert = fits.CompImageHDU(self.uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            else:
                hduUncert = fits.ImageHDU(self.uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            hdus.append(hduUncert)

        if hdu_hits and self.hits is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.hits, "shape"):
                raise ValueError("only a numpy.ndarray hits can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            if compress:
                hduHits = fits.CompImageHDU(self.hits.astype(np.uint32), name=hdu_hits)
            else:
                hduHits = fits.ImageHDU(self.hits.astype(np.uint32), name=hdu_hits)
            hdus.append(hduHits)

        if hdu_flags and self.flags:
            raise NotImplementedError("adding the flags to a HDU is not supported at this time.")

        hdulist = fits.HDUList(hdus)

        return hdulist


class IdiaMixinClass(object):
    """Add possibility to save cube to IDIA hdf5 format"""

    def mip(self, axes=None, level=2, attr="data"):
        """Return a 'Multum In Parvo' of a given level on the data along given axes.

        Parameters
        ----------
        axes: tuple
            the two axis to use (default last 2)
        level: int (power of 2)
            the level of MIP
        attr: str
            the attribute to be used, default 'data'
        """
        if not np.log2(level).is_integer():
            raise ValueError(f"level must be a power of 2 ({level})")

        shape = self.shape

        if axes is None:
            axes = [len(shape) - 2, len(shape) - 1]

        axes = np.sort(axes)
        mip_axes = [i in axes for i in range(len(shape))]

        # nearest multiple of level
        padded_shape = (np.ceil(np.array(shape) / level) * level).astype(int)
        npad = np.where(mip_axes, padded_shape - shape, 0).tolist()
        new_shape = np.where(mip_axes, padded_shape, shape).tolist()

        data = np.pad(getattr(self, attr), list(zip(repeat(0), npad)), mode="edge")

        for axis in axes[::-1]:
            new_shape[axis] //= level
            new_shape = np.insert(new_shape, axis + 1, level)

        mip_data = data.reshape(new_shape).mean(axis=tuple(axes + np.array([1, 2])))
        return mip_data

    def get_axis_from_repr(self, plane: str) -> tuple:
        """Retrieve data axis from representation

        DATA with 4 dimensions can be represented by XYZW
        (stored as WZYX numpy array) celestial are often the XY coordinates,
        this helper function can retrieve the corresponding numpy axis

        Parameters
        ----------
        plane: str
            the requested plane (e,g 'XY', 'XYZ', 'Z')

        Returns
        -------
        axis: tuple
            the corresponding numpy axis in data
        """
        data_repr = "XYZ"[: len(self.shape)]
        if len(self.shape) > 3:
            data_repr += "".join(chr(item) for item in ord("W") - np.arange(0, len(self.shape) - 3))

        # in numpy order
        data_repr = data_repr[::-1]

        # Return the requested axis
        return tuple(data_repr.find(item) for item in plane)

    def to_dset(  # noqa: C901
        self,
        dset_mask="MASK",
        dset_uncertainty="UNCERT",
        dset_hits="HITS",
        dset_flags=None,
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
        planes=["XY", "XYZ", "Z"],
        percentiles_num=100,
        mip_levels=None,
        permuted=None,
        dtype="float32",
    ):
        """Create the IDIA HDF5 datasets and attributes from a FTSData object.

        Parameters
        ----------
        dset_mask, dset_uncertainty, dset_flags, dset_hits : str or None, optional
            If it is a string append this attribute to the dataset dictionnary as
            `~numpy.ndarray`.
            Flags are not supported at this time. If ``None`` this attribute
            is not appended.
            Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty, ``'HITS'`` for hits and
            ``None`` for flags.
        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.
        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.
        planes: list of str
            list of planes on which we retrieve statistic, default ('XY', 'XYZ', 'Z'). See Notes
        percentiles_num: int
            number of percentiles for the returned statistic, default 100
        mip_levels: list of int, optionnal
            list of 'Multum In Parvo' level to be returned, must be of power of 2, default None
        permuted: list of str, optionnal
            list of permuttation plane to be returned, default None
        dtype: str
            dataset dtype, default float32

        Raises
        ------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        NotImplementedError
            Saving flags is not supported.

        Returns
        -------
        dsets : dict
            the dictionnary to be saved
        attrs : dict
            the dictionnary of attribute to be added

        Notes
        -----
        - the list of planes or permutation follows the representation XYZW for a 4th dimension dataset
          (saved as numpy array WZYX) and the 'XY' plane corresponds to the first two axes (last two numpy axes)
        - CARTA only supports float32 dataset
        """
        # Based on https://github.com/idia-astro/hdf5converters/wiki/HDF5-Image-Schema
        dsets = {}
        attrs = {}

        history = None
        comment = None

        if isinstance(self.meta, fits.Header):
            # Remove history and header from fits header
            header = self.meta.copy()
            if "history" in header:
                history = deepcopy(header["history"])
                del header["history"]
            if "comment" in header:
                comment = deepcopy(header["comment"])
                del header["comment"]
        else:
            header = deepcopy(self.meta)
            history = header.pop("history", None)
            comment = header.pop("comment", None)

            # Based on nikamap.utils.meta_to_header
            dummy_header = fits.Header()
            for key, value in header.items():
                if key in ["history", "comment", "HISTORY", "COMMENT"]:
                    continue

                if isinstance(value, (u.Quantity)) or callable(value):
                    value = str(value)

                if not isinstance(
                    value, (int, float, str, complex, np.floating, np.integer, np.complexfloating, np.bool_)
                ):
                    value = json.dumps(value)

                _insert_in_metadata_fits_safe(dummy_header, key, value)

            header = dummy_header

        if self.unit is not None and self.unit is not u.dimensionless_unscaled:
            header["bunit"] = self.unit.to_string()

        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header(relax=wcs_relax)
            header.extend(wcs_header, useblanks=False, update=True)

        # Add NAXIS keywords
        header.update(dict((f"NAXIS{i}", naxis) for i, naxis in enumerate(self.shape[::-1], 1)))

        # CARTA required
        header = dict(header)
        header["SIMPLE"] = "required by CARTA..."
        header["EXTNAME"] = "DATA"
        for key, value in header.items():
            if isinstance(value, str):
                header[key] = np.bytes_(value)

        if comment is not None:
            dsets["COMMENT"] = list(comment)
        if history is not None:
            dsets["HISTORY"] = list(history)

        # DATA is XYZW (store as WZYX) celestial are often the XY coordinates
        dsets["DATA"] = self.data.astype(dtype)
        attrs["0"] = header

        if dset_mask and self.mask is not None:
            if not hasattr(self.mask, "shape"):
                raise ValueError("only a numpy.ndarray mask can be saved.")
            dsets[dset_mask] = self.mask.astype(np.uint8)

        if dset_hits and self.hits is not None:
            if not hasattr(self.hits, "shape"):
                raise ValueError("only a numpy.ndarray hits can be saved.")
            dsets[dset_hits] = self.hits.astype(np.uint32)

        if dset_uncertainty and self.uncertainty is not None:
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError("only uncertainties of type {} can be saved.".format(_known_uncertainties))

            if hasattr(self.uncertainty, "unit") and self.uncertainty.unit is not None and self.unit is not None:
                if not _uncertainty_unit_equivalent_to_parent(uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        "saving uncertainties with a unit that is not "
                        "equivalent to the unit from the data unit is not "
                        "supported."
                    )

            dsets[dset_uncertainty] = self.uncertainty.array.astype(dtype)
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]
            attrs[dset_uncertainty] = {key_uncertainty_type: uncertainty_name}

        if dset_flags and self.flags:
            raise NotImplementedError("adding the flags to a dataset is not supported at this time.")

        # Add Statistics
        percentiles_ranks = np.linspace(0, 100, percentiles_num)

        dsets["PERCENTILE_RANKS"] = percentiles_ranks

        statistics = {}
        for plane in planes:
            plane_statistics = {}
            axis = self.get_axis_from_repr(plane)
            # Compute all statistics
            operations = {
                "MAX": np.nanmax,
                "MIN": np.nanmin,
                "SUM": np.sum,
                "SUM_SQ": lambda x, **kwargs: np.sum(x * x, **kwargs),
                "NAN_COUNT": lambda x, **kwargs: np.sum(np.isnan(x), **kwargs),
                "PERCENTILES": lambda x, **kwargs: np.moveaxis(np.percentile(x, percentiles_ranks, **kwargs), 0, -1),
            }
            # TODO: Add Histograms
            for key, operator in operations.items():
                plane_statistics[key] = operator(self.data, axis=axis).astype(dtype)

            statistics[plane] = plane_statistics

        dsets["Statistics"] = statistics

        # Add Permuted datasets
        if permuted is not None:
            raise NotImplementedError("permutted dataset are not implemented yet")
            # DATA is XYZW (store as WZYX) celestial are often the XY coordinates
            permuteddata = {}
            dsets["PermutedData"] = permuteddata

        # Add MIPMaps
        if mip_levels is not None:
            # MIPS maps only on 'DATA'
            mipmaps = {}
            for level in mip_levels:
                mips_data = self.mip(axes=self.get_axis_from_repr("XY"), level=level)
                mipmaps[f"DATA_XY_{level}"] = mips_data.astype(dtype)

            dsets["MipsMaps"] = {"DATA": mipmaps}

        return dsets, attrs

