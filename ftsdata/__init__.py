"""ftsdata: FTS data containers, transforms, I/O and co-addition.

This package used to be the single ``kidsdata.ftsdata`` module.  The
interferogram base class is :class:`SingleSidedOpdData`; ``OpdData`` is the
generic entry point when the OPD geometry is unknown, and :class:`FTSData` is
kept as a retro-compatibility layer.
"""

import logging

from .version import version as __version__

from ._utils import (
    FITS_HEADER_COMMENT,
    _import_from,
    _insert_in_metadata_fits_safe,
    available_memory,
    cpu_count,
    elevation_to_airmass,
    forman,
    has_axis_type,
    is_itg,
    is_spectral,
    merge_metas,
    safe_map,
)
from .transforms import (
    HAVE_DASK,
    _apply_blockwise_dask,
    _apply_blockwise_threads,
    _to_itg_block,
    _to_spectra_doublesided_block,
    _to_spectra_onesided_block,
    apply_blockwise,
    get_ils_kernel,
    get_itg_binning_correction,
)
from .mixins import (
    HDUMixinClass,
    HitsMixinClass,
    IdiaMixinClass,
    OpdAxisMixinClass,
    SpectralAxisMixinClass,
    WeightMixinClass,
)
from .core import (
    DoubleSidedOpdData,
    FTSData,
    OneSidedOpdData,
    SingleSidedOpdData,
    SpecData,
    _classify_opd,
)

# Importing these modules also registers the FITS/HDF5 readers and writers
# with astropy's unified I/O registry.
from .io import (
    fits_ftsdata_reader,
    fits_ftsdata_writer,
    hdf5_ftsdata_reader,
    hdf5_ftsdata_writer,
    is_idia_hdf5,
)
from .coadd import (
    _ftsdata_average_worker,
    _ftsdata_generator,
    _get_ftsdata,
    ftsdata_average,
    plot_cube_preview,
)

logger = logging.getLogger("FTSData")

# Generic entry point: when the OPD geometry is unknown, ``OpdData`` is the
# name to use.  It is an alias of the base class, so ``from_array`` and
# ``read`` keep dispatching to the geometry-specific subclass.
OpdData = SingleSidedOpdData

__all__ = [
    # Version
    "__version__",
    # Containers
    "SpecData",
    "SingleSidedOpdData",
    "DoubleSidedOpdData",
    "OneSidedOpdData",
    "FTSData",
    "OpdData",
    # Mixins
    "OpdAxisMixinClass",
    "SpectralAxisMixinClass",
    "WeightMixinClass",
    "HitsMixinClass",
    "HDUMixinClass",
    "IdiaMixinClass",
    # Transforms
    "apply_blockwise",
    "get_ils_kernel",
    "get_itg_binning_correction",
    # Metadata / axis helpers
    "forman",
    "has_axis_type",
    "is_spectral",
    "is_itg",
    "merge_metas",
    # I/O
    "fits_ftsdata_reader",
    "fits_ftsdata_writer",
    "hdf5_ftsdata_reader",
    "hdf5_ftsdata_writer",
    "is_idia_hdf5",
    # Co-addition / preview
    "plot_cube_preview",
    "ftsdata_average",
]
