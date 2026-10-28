# ftsdata

Processed interferogram and spectrum data cube class.

## Class hierarchy

`SingleSidedOpdData` is the shared base for interferogram cubes.  It is split
into two geometry-specific subclasses, chosen automatically by
`SingleSidedOpdData.from_array` and `SingleSidedOpdData.read`:

| Class | OPD geometry | Inversion |
|-------|--------------|-----------|
| `DoubleSidedOpdData` | symmetric, ZPD centred | direct FFT (`to_spectra(apodization=...)`) |
| `OneSidedOpdData` | ZPD at the start | real inverse FFT (`to_spectra(apodization=...)`) |
| `SingleSidedOpdData` | asymmetric | optional phase correction, then one-sided inversion |

`FTSData` remains a retro-compatibility class inheriting from both
`SingleSidedOpdData` and `SpecData`; it keeps the generic
(geometry-agnostic) `to_spectra`.

```{note}
`OpdData` is kept as a backward-compatible alias of `SingleSidedOpdData`.
```

## `FTSData`

```
class FTSData(SingleSidedOpdData, SpecData)
```

Stores a processed KIDS FTS data product: either a raw interferogram cube or a calibrated spectrum cube, represented as an astropy `NDDataArray` backed by a FITS file.

### WCS axes

Up to four axes are supported, identified by `CTYPE`:

| CTYPE | UCD | Axis |
|-------|-----|------|
| `OPD` | `pos.opd` | Interferogram optical path difference (mm) |
| `em.freq` | `em.freq` | Frequency (GHz) after FFT |
| `OLON` | `pos.offset.lon` | Sky longitude offset |
| `OLAT` | `pos.offset.lat` | Sky latitude offset |

### Key class methods / functions

#### `has_axis_type(filename, axis_type="em.freq")` → `bool`

Check whether a FITS file has a WCS axis of the given physical type.

#### `is_spectral(filename)` → `bool`

`True` if the file contains a frequency axis (i.e. already FFT'd to spectrum).

#### `is_itg(filename)` → `bool`

`True` if the file contains an OPD axis (i.e. raw interferogram cube).

### Apodisation windows

The module provides standard and instrument-specific window functions for use before FFT:

| Function | Description |
|----------|-------------|
| `forman(M)` | Forman window (from Spencer 2005, Herschel/SPIRE heritage). Defined as `(1 - ((n - M/2)/M)²)²`. |
| Standard scipy windows | `numpy.hanning`, `scipy.signal.windows.*`, etc. |

The Forman window is the recommended default for KISS/CONCERTO data.

### FITS metadata helper

#### `_insert_in_metadata_fits_safe(meta, key, value)`

Insert a key/value pair into FITS metadata, handling the 8-character key and 72-character value length limitations of the FITS standard by splitting long values into `HIERARCH` or `CONTINUE` cards.

### Pipeline integration

`FTSData` objects are the output of `KissSpectroscopy.get_spectrum`. They can be:
- Written to FITS: `fts.write("output.fits")`
- Read back: `fts = FTSData.read("output.fits")`
- Stacked/averaged over multiple scans.
- Used as input to the forward model for calibration.
