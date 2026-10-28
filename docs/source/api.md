# API Reference

`OpdData` is the generic entry point when the OPD geometry is unknown.  It is
an alias of `SingleSidedOpdData`, so `OpdData.from_array` and `OpdData.read`
inspect the OPD axis and return the matching subclass
(`DoubleSidedOpdData`, `OneSidedOpdData` or `SingleSidedOpdData`).  Direct
construction, `OpdData(...)`, does not dispatch: use `from_array` or `read`
when the geometry is not known in advance.  `FTSData` remains the
retro-compatibility class.

```{eval-rst}
.. currentmodule:: ftsdata

.. autosummary::
   :toctree: gen_api
   :nosignatures:

   FTSData
   SpecData
   SingleSidedOpdData
   DoubleSidedOpdData
   OneSidedOpdData
   apply_blockwise
   get_ils_kernel
   get_itg_binning_correction
   forman
   has_axis_type
   is_spectral
   is_itg
   merge_metas
   fits_ftsdata_reader
   fits_ftsdata_writer
   hdf5_ftsdata_reader
   hdf5_ftsdata_writer
   is_idia_hdf5
   plot_cube_preview
   ftsdata_average
```
