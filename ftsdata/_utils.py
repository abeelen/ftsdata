"""Small, self-contained helpers shared by the ftsdata package.

Nothing here imports :mod:`kidsdata` at module level.  Where a kidsdata
equivalent exists (currently only :func:`kidsdata.utils.available_memory`,
which knows about SLURM allocations), it is used lazily when available and a
local fallback is used otherwise.
"""

import importlib
import logging
import multiprocessing
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS
from astropy.wcs.wcsapi.fitswcs import CTYPE_TO_UCD1_CUSTOM

logger = logging.getLogger("FTSData")

FITS_HEADER_COMMENT = {
    "LASER_SHIFT": "[sample] Shift between kids and laser timelines",
    "SITELAT": "[deg] Site latitude",
    "SITEELEV": "[m] Site elevation",
    "DIAMETER": "[m] Telescope diameter",
    "OBSTYPE": "Observation Type",
    "EXPTIME": "[s] Exposure time",
    "POSITION_SHIFT": "[sample] Shift between Kids and Telescope timelines",
    "NKIDS": "Total number of Kids",
    "NINT": "Total number of blocs",
    "NSAMPLE": "Total number of sample",
    "NPTINT": "Number of points per bloc",
    "USED_KIDS": "Number of projected Kids",
    "INTTIME": "[s] Integration Time per sample",
    "MED TAU": "Median zenith tau",
    "MED PWV": "[mm] Median PWV from telescope",
}


def _import_from(attribute, module=None):
    """Import an attribute from a string.

    Parameters
    ----------
    attribute : str
        the name of the attribute to import
    module : str, optional
        the module to import from

    Notes
    -----
    if no module is given, the function tries to infer one from the attribute
    name
    """
    if module is None or "." in attribute:
        *module, attribute = attribute.split(".")
        module = ".".join(module)

    if module == "":
        raise ValueError("Can not determine module")

    module = importlib.import_module(module)
    return getattr(module, attribute)


def cpu_count():
    """Proper cpu count on a SLURM cluster."""
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    return ncpus


def available_memory():
    """Return the available memory in bytes.

    Use :func:`kidsdata.utils.available_memory` when kidsdata is installed,
    because it knows about SLURM cgroups.  That helper assumes a SLURM
    allocation and raises ``KeyError: 'SLURM_JOB_ID'`` elsewhere, so fall back
    to the memory visible to this process.  The fallback keeps the blockwise
    transforms usable on a laptop or a login node without kidsdata.
    """
    try:
        from kidsdata.utils import available_memory as _kidsdata_available_memory
    except ImportError:
        _kidsdata_available_memory = None

    memory = None
    if _kidsdata_available_memory is not None:
        try:
            memory = _kidsdata_available_memory()
        except (KeyError, TypeError, AttributeError, OSError, ValueError):
            memory = None

    if memory is None or memory <= 0:
        try:
            import psutil

            memory = psutil.virtual_memory().available
        except ImportError:
            memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    return memory


def elevation_to_airmass(elevation, formula="PlaneParallel"):
    """Transform elevation to airmass.

    Implement the methods of [1]_

    Parameters
    ----------
    elevation : float or array_like
        the elevation to compute airmass, in degree
    formula : str
        the approximation formula to use.

    Returns
    -------
    airmass : float or array_like
        the corresponding airmass

    Notes
    -----
    formula must be in the list ['PlaneParallel', 'YoungAndIrvine1967',
    'Hardie1962', 'Rozenberg1966', 'KastenAndYoung1989', 'Young1994',
    'Pickering2002']

    .. [1] https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    """
    sec = lambda z: 1 / np.cos(z)  # noqa: E731
    z = np.radians(90 - elevation)

    if formula == "PlaneParallel":
        return sec(z)
    elif formula == "YoungAndIrvine1967":
        return sec(z) * (1 - 0.0012 * (sec(z) ** 2 - 1))
    elif formula == "Hardie1962":
        return sec(z) - 0.0018167 * (sec(z) - 1) - 0.002875 * (sec(z) - 1) ** 2 - 0.0008083 * (sec(z) - 1) ** 3
    elif formula == "Rozenberg1966":
        return (np.cos(z) + 0.025 * np.exp(-11 * np.cos(z))) ** (-1)
    elif formula == "KastenAndYoung1989":
        return 1 / (np.cos(z) + 0.50572 * (6.07995 + 90 - np.degrees(z)) ** (-1.6364))
    elif formula == "Young1994":
        return (1.002432 * np.cos(z) ** 2 + 0.148386 * np.cos(z) + 0.0096467) / (
            np.cos(z) ** 3 + 0.149864 * np.cos(z) ** 2 + 0.0102963 * np.cos(z) + 0.000303978
        )
    elif formula == "Pickering2002":
        return 1 / np.sin(np.radians(elevation + 244 / (165 + 47 * elevation**1.1)))

    raise ValueError("Unknown formula")


def safe_map(
    func: Callable[..., Any],
    *iterables: Iterable[Any],
    processes: Optional[int] = None,
    initializer: Optional[Callable[..., Any]] = None,
    initargs: tuple = (),
) -> list:
    """Execute a function over iterables, capturing exceptions.

    This wraps :class:`concurrent.futures.ProcessPoolExecutor` to catch
    exceptions during execution and raise them at the end, preventing silent
    failures.  All errors are collected and reported together with their input
    context.

    Parameters
    ----------
    func : Callable
        Function to execute for each item in iterables
    iterables : Iterable
        Items to process
    processes : int, optional
        Maximum number of worker processes. If None, uses :func:`cpu_count`
    initializer : Callable, optional
        Function to call at worker process startup
    initargs : tuple, optional
        Arguments to pass to initializer

    Returns
    -------
    list
        List of results from successful executions

    Raises
    ------
    ValueError
        If any exception occurred during pool execution, with details of all
        failures
    """
    errors = []
    results = []

    if processes is None:
        processes = cpu_count()

    # Prepare inputs: zip handles both single and multiple iterables uniformly
    inputs = list(zip(*iterables))

    with ProcessPoolExecutor(max_workers=processes, initializer=initializer, initargs=initargs) as pool:
        # Submit all tasks
        futures = {pool.submit(func, *item): (idx, item) for idx, item in enumerate(inputs)}

        # Collect results and errors
        for future in futures:
            idx, item = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                error_info = {
                    "index": idx,
                    "input": item,
                    "exception": e,
                    "exception_type": type(e).__name__,
                    "error_message": str(e),
                }
                errors.append(error_info)
                logger.error(error_info["error_message"])

    # Raise if any errors occurred
    if errors:
        error_summary = "\n".join(
            [
                f"  - Index {err['index']}, Input {err['input']}: {err['exception_type']}: {err['error_message']}"
                for err in errors
            ]
        )
        raise ValueError(
            f"Errors occurred in {len(errors)}/{len(inputs)} tasks:\n{error_summary}\n"
            f"Possible silent OOM kill if no specific errors listed."
        )

    # Sort results by index and return values only
    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


# Add Custom CTYPE for FTSData.  This must run before any container is built:
# the WCS physical-type detection (pos.opd) relies on it.
CTYPE_TO_UCD1_CUSTOM.append({"OPD": "pos.opd", "OLON": "pos.offset.lon", "OLAT": "pos.offset.lat"})


def has_axis_type(filename, axis_type="em.freq", **kwargs):
    header = fits.getheader(filename, **kwargs)
    physical_type = WCS(header).world_axis_physical_types
    return axis_type in physical_type


def is_spectral(filename, **kwargs):
    return has_axis_type(filename, axis_type="em.freq", **kwargs)


def is_itg(filename, **kwargs):
    return has_axis_type(filename, axis_type="pos.opd", **kwargs)


def forman(M):
    """Return Forman window.

    The Forman window is defined in (E-4) [1]_.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    numpy.bartlett, numpy.blackman, numpy.hamming, numpy.kaiser, numpy.hanning

    References
    ----------
    .. [1] Spencer, L.D., (2005) Spectral Characterization of the Herschel SPIRE
          Photometer, 2005MsT..........1S
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return (1 - ((n - M / 2) / M) ** 2) ** 2


# Thanks astropy/NDDATA...
# from from CCDData._insert_in_metadata_fits_safe
def _insert_in_metadata_fits_safe(meta, key, value):
    """
    Insert key/value pair into metadata in a way that FITS can serialize.

    Parameters
    ----------
    key : str
        Key to be inserted in dictionary.

    value : str or None
        Value to be inserted.

    Notes
    -----
    This addresses a shortcoming of the FITS standard. There are length
    restrictions on both the ``key`` (8 characters) and ``value`` (72
    characters) in the FITS standard. There is a convention for handling
    long keywords and a convention for handling long values, but the
    two conventions cannot be used at the same time.

    This addresses that case by checking the length of the ``key`` and
    ``value`` and, if necessary, shortening the key.
    """
    if len(key) > 8 and isinstance(value, str) and len(value) > 72:
        short_name = key[:8]
        meta[f"HIERARCH {key.upper()}"] = (
            short_name,
            f"Shortened name for {key}",
        )
        meta[short_name] = value
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=VerifyWarning)
            meta[key] = value



def merge_metas(meta_list):
    meta = defaultdict(list)
    history = defaultdict(list)
    comment = defaultdict(list)
    for _meta in meta_list:
        for key, value in _meta.items():
            if key not in [
                "HISTORY",
                "COMMENT",
                "SIMPLE",
                "BITPIX",
                "NAXIS",
                "EXTEND",
                "NAXIS1",
                "NAXIS2",
                "NAXIS3",
                "PCOUNT",
                "GOUNT",
            ]:
                meta[key].append(value)
            elif key == "HISTORY":
                history[key].append(value)
            elif key == "COMMENT":
                comment[key].append(value)

    for key in meta:
        if len(set(meta[key])) == 1:
            meta[key] = meta[key][0]
        else:
            meta[key] = list(set(meta[key]))

    return meta
