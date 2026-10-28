# ftsdata

`ftsdata` provides the Fourier-transform spectrometer data containers,
transforms, FITS/HDF5 I/O and co-addition routines used by the CONCERTO/KIDS
reduction.  It used to be the single `kidsdata.ftsdata` module and is now a
standalone distribution.

The public API is unchanged:

```python
from ftsdata import FTSData, OpdData, SpecData, ftsdata_average
```

## Installation

From this repository:

```bash
python -m pip install -e ./packages/ftsdata
```

Optional features are grouped in extras:

```bash
python -m pip install -e "./packages/ftsdata[all]"   # h5py, dask, matplotlib, psutil
```

Only NumPy, SciPy and Astropy are required for the containers and transforms.
`h5py` is needed for HDF5 I/O, `dask` for dask-backed blockwise transforms,
`matplotlib` for `plot_cube_preview`, and `psutil` improves memory detection.

## Relationship with kidsdata

`ftsdata` does not import `kidsdata` at module import time, so it can be used
on its own.  When the `kidsdata` distribution is installed, two integrations
remain available lazily:

- `available_memory()` uses `kidsdata.utils.available_memory` when it can,
  which knows about SLURM allocations.
- `SpecData.correct_opacity()` uses `kidsdata.forward_model.ATM` by default.
  Without kidsdata, pass a compatible model explicitly:

  ```python
  cube.correct_opacity(elevation=60, pwv=1.0, atm=my_atm_model)
  ```

Existing code keeps working through the compatibility shim
`kidsdata.ftsdata`:

```python
from kidsdata.ftsdata import FTSData
```

## Tests

```bash
python -m pytest packages/ftsdata/tests
```

## Documentation

The Sphinx sources are in `packages/ftsdata/docs/`:

```bash
python -m pip install -e "./packages/ftsdata[docs]"
python -m sphinx -b html packages/ftsdata/docs/source packages/ftsdata/docs/_build/html
```
