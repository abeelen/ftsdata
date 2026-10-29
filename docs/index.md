# ftsdata documentation

`ftsdata` provides the Fourier-transform spectrometer data containers,
transforms, FITS/HDF5 I/O and co-addition routines used by the CONCERTO/KIDS
reduction.  It is a standalone distribution: it does not import `kidsdata` at
module import time, but keeps optional integrations when `kidsdata` is
installed.

```{eval-rst}
.. plot:: _static/plot_opd_geometries.py
```

```{toctree}
:maxdepth: 2

usage
api
min_versions
```

## Installation

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[all]"
```

## Import paths

```python
from ftsdata import FTSData, SpecData, ftsdata_average
```

Existing code can keep using the compatibility shim:

```python
from kidsdata.ftsdata import FTSData
```

## Building these docs

```bash
python -m pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```
