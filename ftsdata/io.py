"""FITS and HDF5 I/O, plus astropy registry wiring, for ftsdata."""

import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
from astropy.io import fits, registry
from astropy.nddata import StdDevUncertainty
from astropy.nddata.ccddata import _generate_wcs_and_update_header, _unc_name_to_cls

from ._hdf5 import dict_to_hdf5, open_hdf5
from .core import (
    DoubleSidedOpdData,
    FTSData,
    OneSidedOpdData,
    SingleSidedOpdData,
    SpecData,
    _classify_opd,
)

logger = logging.getLogger("FTSData")

def fits_ftsdata_writer(
    fts_data,
    filename,
    hdu_mask="MASK",
    hdu_uncertainty="UNCERT",
    hdu_hits="HITS",
    hdu_flags=None,
    key_uncertainty_type="UTYPE",
    compress=False,
    **kwd,
):
    """
    Write FTSData object to FITS file.

    Parameters
    ----------
    filename : str
        Name of file.
    hdu_mask, hdu_uncertainty, hdu_hits, hdu_flags : str or None, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty ``'HITS'`` for hits and
        ``None`` for flags.
    key_uncertainty_type : str, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.
    compress : bool, optional
        Use tiled compression ImageHDU for the data. Default is ``False``.
    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`

    Raises
    ------
    ValueError
        - If ``self.mask`` is set but not a `numpy.ndarray`.
        - If ``self.uncertainty`` is set but not a
          `~astropy.nddata.StdDevUncertainty`.
        - If ``self.uncertainty`` is set but has another unit then
          ``self.data``.

    NotImplementedError
        Saving flags is not supported.
    """
    hdu = fts_data.to_hdu(
        hdu_mask=hdu_mask,
        hdu_uncertainty=hdu_uncertainty,
        hdu_hits=hdu_hits,
        key_uncertainty_type=key_uncertainty_type,
        hdu_flags=hdu_flags,
        compress=compress,
    )
    hdu.writeto(filename, **kwd)

def fits_ftsdata_reader(
    filename,
    hdu=0,
    unit=None,
    hdu_mask="MASK",
    hdu_uncertainty="UNCERT",
    hdu_hits="HITS",
    hdu_flags=None,
    key_uncertainty_type="UTYPE",
    cls=None,
    **kwd,
):
    """
    Generate a FTS Data object from a FITS file.

    Parameters
    ----------
    filename : str
        Name of fits file.

    hdu : int, str, tuple of (str, int), optional
        Index or other identifier of the Header Data Unit of the FITS
        file from which FTSData should be initialized. If zero and
        no data in the primary HDU, it will search for the first
        extension HDU with data. The header will be added to the primary HDU.
        Default is ``0``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.
        Default is ``None``.

    hdu_uncertainty : str or None, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty of the CCDData is ``None``.
        Default is ``'UNCERT'``.

    hdu_mask : str or None, optional
        FITS extension from which the mask should be initialized. If the
        extension does not exist the mask of the CCDData is ``None``.
        Default is ``'MASK'``.

    hdu_flags : str or None, optional
        Currently not implemented.
        Default is ``None``.

    key_uncertainty_type : str, optional
        The header key name where the class name of the uncertainty  is stored
        in the hdu of the uncertainty (if any).
        Default is ``UTYPE``.

    kwd :
        Any additional keyword parameters are passed through to the FITS reader
        in :mod:`astropy.io.fits`; see Notes for additional discussion.

    Notes
    -----
    FITS files that contained scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.
    """
    unsupport_open_keywords = {
        "do_not_scale_image_data": "Image data must be scaled.",
        "scale_back": "Scale information is not preserved.",
    }
    for key, msg in unsupport_open_keywords.items():
        if key in kwd:
            prefix = f"unsupported keyword: {key}."
            raise TypeError(" ".join([prefix, msg]))
    with fits.open(filename, **kwd) as hdus:
        hdr = hdus[hdu].header

        if hdu_uncertainty is not None and hdu_uncertainty in hdus:
            unc_hdu = hdus[hdu_uncertainty]
            stored_unc_name = unc_hdu.header.get(key_uncertainty_type, "None")
            # For compatibility reasons the default is standard deviation
            # uncertainty because files could have been created before the
            # uncertainty type was stored in the header.
            unc_type = _unc_name_to_cls.get(stored_unc_name, StdDevUncertainty)
            uncertainty = unc_type(unc_hdu.data)
        else:
            uncertainty = None

        if hdu_mask is not None and hdu_mask in hdus:
            # Mask is saved as uint but we want it to be boolean.
            mask = hdus[hdu_mask].data.astype(bool)
        else:
            mask = None

        if hdu_flags is not None and hdu_flags in hdus:
            raise NotImplementedError("loading flags is currently not supported.")

        if hdu_hits is not None and hdu_hits in hdus:
            hits = hdus[hdu_hits].data.astype(int)
        else:
            hits = None

        # search for the first instance with data if
        # the primary header is empty.
        if hdu == 0 and hdus[hdu].data is None:
            for i in range(len(hdus)):
                if hdus.info(hdu)[i][3] == "ImageHDU" and hdus.fileinfo(i)["datSpan"] > 0:
                    hdu = i
                    comb_hdr = hdus[hdu].header.copy()
                    # Add header values from the primary header that aren't
                    # present in the extension header.
                    comb_hdr.extend(hdr, unique=True)
                    hdr = comb_hdr
                    logger.info(f"first HDU with data is extension {hdu}.")
                    break

        if "bunit" in hdr:
            fits_unit_string = hdr["bunit"]
            # patch to handle FITS files using ADU for the unit instead of the
            # standard version of 'adu'
            if fits_unit_string.strip().lower() == "adu":
                fits_unit_string = fits_unit_string.lower()
        else:
            fits_unit_string = None

        use_unit = unit or fits_unit_string
        hdr, wcs = _generate_wcs_and_update_header(hdr)

        data = hdus[hdu].data

        # https://github.com/astropy/astropy/issues/4069
        sys_byteorder = (">", "<")[sys.byteorder == "little"]
        if data.dtype.byteorder not in ("=", sys_byteorder):
            logger.info("Swapping byte order")
            data = data.astype(data.dtype.newbyteorder("="))

        resolved_cls = cls or FTSData
        if resolved_cls is SingleSidedOpdData:
            resolved_cls = _classify_opd(wcs, shape=data.shape)

        fts_data = resolved_cls(
            data,
            meta=hdr,
            unit=use_unit,
            mask=mask,
            uncertainty=uncertainty,
            hits=hits,
            wcs=wcs,
        )

    return fts_data

def hdf5_ftsdata_reader(
    filename,
    unit=None,
    dset_mask="MASK",
    dset_uncertainty="UNCERT",
    dset_hits="HITS",
    dset_flags=None,
    key_uncertainty_type="UTYPE",
    cls=None,
):
    """
    Generate an FTSData object from an IDIA HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to read.
    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.
        Default is ``None``.
    dset_mask, dset_uncertainty, dset_flags, dset_hits : str or None, optional
        If it is a string append this attribute to the dataset dictionnary as
        `~numpy.ndarray`.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty, ``'HITS'`` for hits and
        ``None`` for flags.
    key_uncertainty_type : str, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.
    """
    # no memmap for now...
    with open_hdf5(filename, mode="r") as f:
        version = f["0"]

        hdr = fits.Header(
            {k: v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v for k, v in version.attrs.items()}
        )

        if dset_uncertainty is not None and dset_uncertainty in version:
            uncert_dset = version[dset_uncertainty]
            stored_unc_name = uncert_dset.attrs.get(key_uncertainty_type, "None")
            # For compatibility reasons the default is standard deviation
            # uncertainty because files could have been created before the
            # uncertainty type was stored in the header.
            unc_type = _unc_name_to_cls.get(stored_unc_name, StdDevUncertainty)
            uncertainty = unc_type(uncert_dset[:])
        else:
            uncertainty = None

        if dset_mask is not None and dset_mask in version:
            mask = version[dset_mask][:]
        else:
            mask = None

        if dset_flags is not None and dset_flags in version:
            raise NotImplementedError("loading flags is currently not supported.")

        if dset_hits is not None and dset_hits in version:
            hits = version[dset_hits][:]
        else:
            hits = None

        if "bunit" in hdr:
            fits_unit_string = hdr["bunit"]
            # patch to handle FITS files using ADU for the unit instead of the
            # standard version of 'adu'
            if fits_unit_string.strip().lower() == "adu":
                fits_unit_string = fits_unit_string.lower()
        else:
            fits_unit_string = None

        use_unit = unit or fits_unit_string
        hdr, wcs = _generate_wcs_and_update_header(hdr)

        data = version["DATA"][:]

        resolved_cls = cls or FTSData
        if resolved_cls is SingleSidedOpdData:
            resolved_cls = _classify_opd(wcs, shape=data.shape)

        fts_data = resolved_cls(
            data,
            meta=hdr,
            unit=use_unit,
            mask=mask,
            uncertainty=uncertainty,
            hits=hits,
            wcs=wcs,
        )

    return fts_data

def hdf5_ftsdata_writer(
    fts_data,
    filename,
    mode="w",
    file_kwargs=None,
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
    **kwargs,
):
    """
    Write an FTSData object to an IDIA HDF5 file.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file to write.
    mode : str
        the open mode for the h5py.File object, by default 'w'
    file_kwargs, dict, optionnal
        additionnal keyword for h5py.File object, by default None
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
    mip_level: list of int, optionnal
        list of 'Multum In Parvo' level to be returned, must be of power of 2, default None
    permuted: list of str, optionnal
        list of permuttation plane to be returned, default None
    dtype: str
        dataset dtype, default float32
    **kwargs
        additionnal keyword argument for the h5py.Dataset, see Note

    Notes
    -----
    Usual kwargs could be :
    kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    However, the best compromise in perfomance/size on disk is obtained with kwargs=hdf5plugin.Bitshuffle()
    """

    # https://github.com/idia-astro/hdf5converters/wiki/HDF5-Image-Schema
    if file_kwargs is None:
        file_kwargs = {}

    dsets, attrs = fts_data.to_dset(
        dset_mask=dset_mask,
        dset_uncertainty=dset_uncertainty,
        dset_hits=dset_hits,
        dset_flags=dset_flags,
        wcs_relax=wcs_relax,
        key_uncertainty_type=key_uncertainty_type,
        planes=planes,
        percentiles_num=percentiles_num,
        mip_levels=mip_levels,
        permuted=permuted,
        dtype=dtype,
    )

    with open_hdf5(filename, mode=mode, **file_kwargs) as f:
        # Create the dataset
        f.attrs["VERSION"] = "0"
        version = f.create_group("0")
        if "0" in attrs:
            attr = attrs.pop("0")
            version.attrs.update(attr)

        dict_to_hdf5(version, dsets, **kwargs)
        for key, values in attrs.items():
            version[key].attrs.update(values)

def is_idia_hdf5(origin, filepath, *args, **kwargs):
    """
    Determine whether `origin` is a IDIA HDF5 file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a potential HDF5 file.

    Returns
    -------
    is_idia : bool
        Returns `True` if the given file is a IDIA HDF5 file.
    """
    is_idia = isinstance(filepath, (str, Path))
    if filepath is not None:
        is_idia &= filepath.lower().endswith((".hdf5", ".h5"))
        # if is_idia and Path(filepath).exists():
        #     with open_hdf5(filepath, mode="r") as f:
        #         is_idia &= "VERSION" in f.attrs
        #         if is_idia:
        #             version = f.attrs["VERSION"]
        #             is_idia &= str(version) == "0"
        #         if is_idia:
        #             is_idia &= "0" in f
        #             group = f["0"]
        #             is_idia &= "DATA" in group
    return is_idia

with registry.delay_doc_updates(FTSData):
    registry.register_reader("fits", FTSData, fits_ftsdata_reader)
    registry.register_writer("fits", FTSData, fits_ftsdata_writer)
    registry.register_identifier("fits", FTSData, fits.connect.is_fits)
    registry.register_reader("hdf5", FTSData, hdf5_ftsdata_reader)
    registry.register_writer("hdf5", FTSData, hdf5_ftsdata_writer)
    registry.register_identifier("hdf5", FTSData, is_idia_hdf5)

for _cls in (DoubleSidedOpdData, OneSidedOpdData, SingleSidedOpdData):
    with registry.delay_doc_updates(_cls):
        registry.register_reader("fits", _cls, partial(fits_ftsdata_reader, cls=_cls))
        registry.register_writer("fits", _cls, fits_ftsdata_writer)
        registry.register_identifier("fits", _cls, fits.connect.is_fits)
        registry.register_reader("hdf5", _cls, partial(hdf5_ftsdata_reader, cls=_cls))
        registry.register_writer("hdf5", _cls, hdf5_ftsdata_writer)
        registry.register_identifier("hdf5", _cls, is_idia_hdf5)

del _cls

with registry.delay_doc_updates(SpecData):
    registry.register_reader("fits", SpecData, partial(fits_ftsdata_reader, cls=SpecData))
    registry.register_writer("fits", SpecData, fits_ftsdata_writer)
    registry.register_identifier("fits", SpecData, fits.connect.is_fits)
    registry.register_reader("hdf5", SpecData, partial(hdf5_ftsdata_reader, cls=SpecData))
    registry.register_writer("hdf5", SpecData, hdf5_ftsdata_writer)
    registry.register_identifier("hdf5", SpecData, is_idia_hdf5)
