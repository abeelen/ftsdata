"""Minimal HDF5 writing helpers used by the ftsdata I/O module.

These helpers used to live in :mod:`kidsdata.read_kidsdata`.  They are copied
here so ftsdata can be installed and imported without kidsdata; ``h5py`` and
the astropy HDF5 table writer are imported lazily.
"""

import logging
from functools import partial, wraps
from pathlib import Path
from time import sleep

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

logger = logging.getLogger("FTSData")


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] is not tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)


def open_hdf5(filename, *args, **kwargs):
    """Open an HDF5 file, retrying while the file is locked.

    h5py refuses to open a file that is already open in an incompatible mode;
    retry every 10 seconds instead of failing immediately.
    """
    import h5py

    while True:
        try:
            return h5py.File(filename, *args, **kwargs)
        except OSError:
            logger.warning(f"Unable to open {filename}, waiting 10 sec to retry")
            sleep(10)


def filename_or_h5py_file(func=None, mode="r"):
    """Allow a filename/Path or an h5py.File instance as first argument."""
    if func is None:
        return partial(filename_or_h5py_file, mode=mode)

    @wraps(func)
    def wrapper(*args, **kwargs):
        filename = args[0]

        if isinstance(filename, (str, Path)):
            with open_hdf5(filename, mode=mode) as f:
                output = func(f, *args[1:], **kwargs)
        else:
            output = func(filename, *args[1:], **kwargs)

        return output

    return wrapper


@filename_or_h5py_file(mode="a")
def _to_hdf5(parent_group, key, data, **kwargs):  # noqa: C901
    """Save data to hdf5.

    Parameters
    ----------
    parent_group : `h5py.Group`
        the parent group to save the dataset
    key : str
        the name of the dataset
    data : (np.array|astropy.table.Table|int|float|namedtuple|tuple|list|dict)
        the corresponding dataset

    Notes
    -----
    the dataset can only be a combination of numpy arrays, tuple, list or
    dictionaries, and some specific types like
    astropy.table.table.Table astropy.time.core.Time astropy.wcs.wcs.WCS
    astropy.io.fits.header.Header

    Any keyword supported by h5py.create_dataset can be passed, usually :
    kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    """
    if isinstance(data, np.ndarray) and not isinstance(data, np.ma.MaskedArray):
        # numpy arrays
        if key in parent_group:
            del parent_group[key]
        # Special case for unicode strings
        if "U" in data.dtype.str:
            data = data.astype("S")
        return parent_group.create_dataset(key, data=data, **kwargs)

    elif isinstance(data, Table):
        # astropy table
        from astropy.io.misc.hdf5 import write_table_hdf5

        if key in parent_group:
            del parent_group[key]
        parent_group.create_group(key)

        write_table_hdf5(data, parent_group, path=key, append=True, overwrite=True, serialize_meta=True)
        dset = parent_group[key]
        dset.attrs["type"] = "astropy.table.table.Table"
        return dset

    elif isinstance(data, Time):
        # astropy Time data
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_dataset(key, data=data.to_value("mjd"), **kwargs)
        dset.attrs["type"] = "astropy.time.core.Time"
        dset.attrs["scale"] = data.scale
        dset.attrs["format"] = data.format
        return dset

    elif isinstance(data, (str, int, np.int32, np.int64, float, np.float32, np.float64, bool)):
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_dataset(key, data=data)
        return dset

    # list of same type and same size
    elif (
        isinstance(data, list)
        and not isinstance(data[0], dict)
        and not isinstance(data[0], tuple)
        and all([isinstance(_data, type(data[0])) for _data in data[1:]])
        and (np.isscalar(data[0]) or all([len(_data) == len(data[0]) for _data in data[1:]]))
    ):
        import h5py

        if key in parent_group:
            del parent_group[key]
        if isinstance(data[0], (str, np.str_)):
            # strings:
            dt = h5py.string_dtype()
            dset = parent_group.create_dataset(key, data=np.array(data, dtype="S"), dtype=dt)
        else:
            dset = parent_group.create_dataset(key, data=np.array(data))
        dset.attrs["type"] = "list-array"
        return dset

    elif data is None:
        if key in parent_group.attrs:
            del parent_group.attrs[key]
        if key in parent_group:
            del parent_group[key]
        return parent_group.create_dataset(key, dtype="f")

    elif isinstance(data, WCS):
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_dataset(key, data=data.to_header_string(), **kwargs)
        dset.attrs["type"] = "astropy.wcs.wcs.WCS"

    elif isinstance(data, fits.Header):
        if key in parent_group:
            del parent_group[key]
        dset = parent_group.create_dataset(key, data=data.tostring())
        dset.attrs["type"] = "astropy.io.fits.header.Header"

    elif isinstance(data, (tuple, list, dict, np.ma.MaskedArray)):
        if key not in parent_group:
            sub_group = parent_group.create_group(key)
            sub_group.attrs["type"] = type(data).__name__
        else:
            sub_group = parent_group[key]
    else:
        raise TypeError("Can not handle type of {} :  {}".format(key, type(data)))

    if isnamedtupleinstance(data):
        _to_hdf5(sub_group, str(type(data)), data._asdict(), **kwargs)
        sub_group.attrs["type"] = "namedtuple"
    elif isinstance(data, (tuple, list)):
        for i, item in enumerate(data):
            _to_hdf5(sub_group, str(i), item, **kwargs)
    elif isinstance(data, dict):
        # Force string keys.... will cause problem for KidsData__dataRg
        for _key in data:
            _to_hdf5(sub_group, str(_key), data[_key], **kwargs)
    elif isinstance(data, np.ma.MaskedArray):
        sub_group.attrs["fill_value"] = getattr(data, "fill_value")
        for _key in ["data", "mask"]:
            _to_hdf5(sub_group, _key, getattr(data, _key), **kwargs)

    return None


@filename_or_h5py_file(mode="a")
def dict_to_hdf5(parent_group, data, **kwargs):  # noqa: C901
    """Save a python dictionary to hdf5.

    Parameters
    ----------
    parent_group : `h5py.Dataset`
        the dataset or filename
    data : dict
        the dictionary to save

    Notes
    -----
    the dictionary can only contain numpy arrays, tuple, list or dictionaries,
    and some specific types like astropy.table.table.Table
    astropy.time.core.Time astropy.wcs.wcs.WCS astropy.io.fits.header.Header

    Any keyword supported by h5py.create_dataset can be passed, usually :
    kwargs={'chunks': True, 'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    """
    assert isinstance(data, dict)
    for key, item in data.items():
        _to_hdf5(parent_group, key, item, **kwargs)
