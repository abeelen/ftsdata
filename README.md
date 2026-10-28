# ftsdata

[![pipeline status](https://gitlab.lam.fr/KISS/ftsdata/badges/main/pipeline.svg)](https://gitlab.lam.fr/KISS/ftsdata/-/pipelines)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`ftsdata` is the Fourier-transform spectrometer (FTS) data toolkit for the
CONCERTO and KIDS reductions.  It provides OPD- and frequency-aware data
containers, interferogram/spectrum transforms, FITS and IDIA HDF5 I/O, and
inverse-variance co-addition of spectral cubes.

The package used to be the single `kidsdata.ftsdata` module.  It is now an
independent distribution, and existing code keeps working through the
`kidsdata.ftsdata` compatibility shim.

## Features

- **Data containers** — `SingleSidedOpdData`, `DoubleSidedOpdData`,
  `OneSidedOpdData`, `SpecData`, and the retro-compatible `FTSData`.
  `OpdData` is the geometry-agnostic entry point (`from_array` / `read`).
- **Interferogram → spectrum** — Forman phase correction, apodisation,
  zero-padding and ITG binning correction.
- **Spectrum → interferogram** — `to_itg` for forward modelling.
- **I/O** — FITS products registered with Astropy's unified I/O, plus export to
  the IDIA HDF5 schema used by CARTA.
- **Large cubes** — thread-based or dask-backed blockwise processing.
- **Co-addition** — `ftsdata_average` combines cubes with per-pixel
  inverse-variance weighting.
- **Optional kidsdata integration** — opacity correction can use
  `kidsdata.forward_model.ATM` when kidsdata is installed.

## Support policy

`ftsdata` follows [SPEC 0 — Minimum Supported Dependencies](https://scientific-python.org/specs/spec-0000/)
for Python and core dependency support, with one explicit exception: **Python
3.8 remains supported** for the CONCERTO environment used by `kidsdata`.

On Python 3.8 the package uses the last compatible NumPy, SciPy and Astropy
releases; newer Python versions use the current SPEC 0 floors.  Minimum
versions are raised on minor or major releases, never on patch releases.  The
current floors and supported versions are listed in
[docs/source/min_versions.md](docs/source/min_versions.md).

## Installation

The package is not on PyPI yet; install it from this repository (Python 3.8+):

```bash
git clone git@gitlab.lam.fr:KISS/ftsdata.git
cd ftsdata
python -m pip install -e .
```

Optional features are grouped in extras:

```bash
python -m pip install -e ".[all]"     # h5py, dask, matplotlib, psutil
python -m pip install -e ".[hdf5]"    # HDF5 / IDIA export
python -m pip install -e ".[dask]"    # dask-backed blockwise transforms
python -m pip install -e ".[plot]"    # cube previews
python -m pip install -e ".[docs]"    # Sphinx documentation build
```

Only NumPy, SciPy and Astropy are required for the core containers and
transforms.  `h5py` enables HDF5 I/O, `dask` enables dask-backed transforms,
`matplotlib` enables `plot_cube_preview`, and `psutil` improves memory
detection.

## Quick start

```python
import astropy.units as u
import numpy as np
from ftsdata import SingleSidedOpdData

# 101 OPD samples with a 100 µm step, symmetric around the ZPD
opd = np.arange(-50, 51) * 0.1 * u.mm
data = np.random.default_rng(42).normal(size=(opd.size, 16, 16))

# from_array inspects the OPD geometry and returns the matching subclass
itg = SingleSidedOpdData.from_array(opd, data)
print(type(itg).__name__)          # DoubleSidedOpdData

# Apodise, zero-pad and FFT to a spectral cube
spectrum = itg.to_spectra(apodization=np.hanning, n_pad=3)
print(spectrum.data.shape)         # (nfreq, 16, 16)
print(spectrum.spectral_axis.to("GHz")[:3])

spectrum.write("spectrum.fits", overwrite=True)
```

Read a product back with a compatible class:

```python
from ftsdata import FTSData

cube = FTSData.read("spectrum.fits")
print(cube.data.shape)
```

## Data model

All containers derive from `astropy.nddata.NDDataArray` and carry:

| Attribute | Description |
|-----------|-------------|
| `data` | Data cube, normally `(nopd, ny, nx)` or `(nfreq, ny, nx)` |
| `uncertainty` | Astropy uncertainty array |
| `mask` | Boolean bad-pixel mask (`True` = invalid) |
| `wcs` | Full WCS (sky axes plus OPD or frequency) |
| `meta` | Provenance and pipeline parameters |
| `hits` | Number of detector samples per voxel |

| Class | Role |
|-------|------|
| `SingleSidedOpdData` | Base interferogram cube; `from_array` dispatches on OPD geometry |
| `DoubleSidedOpdData` | Symmetric OPD grid with the ZPD centred |
| `OneSidedOpdData` | OPD grid starting at the ZPD; real inverse FFT |
| `SpecData` | Spectrum cube |
| `FTSData` | Retro-compatible class inheriting from the interferogram and spectrum classes |
| `OpdData` | Generic entry point when the OPD geometry is unknown; alias of `SingleSidedOpdData`, so `OpdData.from_array` / `OpdData.read` still dispatch |

## I/O and co-addition

FITS products use the Astropy unified I/O interface:

```python
from ftsdata import FTSData

spectrum.write("cube.fits", overwrite=True)
spectrum.write("cube-compressed.fits", compress=True, overwrite=True)
cube = FTSData.read("cube.fits")
```

For CARTA, export the IDIA HDF5 layout with `to_dset`:

```python
import h5py

dsets, attrs = cube.to_dset(mip_levels=[2, 4, 8], dtype="float32")
with h5py.File("cube.hdf5", "w") as f:
    for key, values in dsets.items():
        f.create_dataset(key, data=values)
    for key, values in attrs.items():
        for name, value in values.items():
            f[key].attrs[name] = value
```

Co-add cubes that share the same WCS grid:

```python
from pathlib import Path
from ftsdata import FTSData, ftsdata_average

cubes = [FTSData.read(path) for path in sorted(Path("products").glob("*_T.fits"))]
combined = ftsdata_average(cubes)
combined.write("combined_T.fits", overwrite=True)
```

## Relationship with kidsdata

`ftsdata` does not import `kidsdata` at module import time, so it can be used
on its own.  When the `kidsdata` distribution is installed, two integrations
become available lazily:

- `available_memory()` uses `kidsdata.utils.available_memory` when available,
  which knows about SLURM allocations.
- `SpecData.correct_opacity()` uses `kidsdata.forward_model.ATM` by default.
  Without kidsdata, pass a compatible model explicitly:

  ```python
  cube.correct_opacity(elevation=60, pwv=1.0, atm=my_atm_model)
  ```

Existing code can keep using the old import path:

```python
from kidsdata.ftsdata import FTSData
```

New code should import from `ftsdata` directly.

## Documentation

The Sphinx sources live in `docs/`:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs/source docs/_build/html
```

The built pages are published by GitLab Pages from the `main` branch.

## Tests

```bash
python -m pip install -e ".[all]" pytest
python -m pytest tests -q
```

The CI pipeline runs the test-suite on several Python versions and builds the
documentation with warnings treated as errors.

## Repository layout

```
ftsdata/            package modules
tests/              pytest suite
docs/               Sphinx sources
pyproject.toml      packaging and tool configuration
.gitlab-ci.yml      tests and GitLab Pages
```

## License

BSD 3-clause.  See [LICENSE](LICENSE).
