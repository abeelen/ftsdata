"""Weighted co-addition of spectral cubes."""

import numpy as np
import numpy.testing as npt
import pytest
from astropy.nddata import InverseVariance, StdDevUncertainty

from ftsdata import FTSData, ftsdata_average


def _make_spectral_cube(wcs, data, hits=None, mask=None, meta=None):
    if hits is None:
        hits = np.ones_like(data)
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)
    if meta is None:
        meta = {
            "NAXIS": 3,
            "NAXIS1": data.shape[0],
            "NAXIS2": data.shape[1],
            "NAXIS3": data.shape[2],
        }
    return FTSData(
        data=data,
        uncertainty=StdDevUncertainty(np.ones_like(data)),
        wcs=wcs,
        hits=hits,
        mask=mask,
        meta=meta,
    )


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_ftsdata_average(gaussian_ds, tmp_path):
    """Average two identical cubes: the weighted mean is the cube itself."""
    wcs = gaussian_ds.to_spectra().wcs
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1025, 2, 2)) ** 2

    paths = []
    for i in range(2):
        cube = _make_spectral_cube(wcs, data.copy())
        path = tmp_path / f"cube_{i}.fits"
        cube.write(path, format="fits", overwrite=True)
        paths.append(str(path))

    averaged = ftsdata_average(paths, min_average_per_cpu=1)

    assert isinstance(averaged, FTSData)
    assert averaged.data.shape == data.shape
    npt.assert_allclose(averaged.data, data, rtol=1e-6, atol=1e-8)
    assert isinstance(averaged.uncertainty, InverseVariance)
    # Both cubes contribute one hit per pixel.
    npt.assert_array_equal(averaged.hits, 2)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_ftsdata_average_accepts_path_objects(gaussian_ds, tmp_path):
    """A list of ``Path`` objects is read lazily, like a list of strings."""
    wcs = gaussian_ds.to_spectra().wcs
    data = np.linspace(1, 2, 1025)[:, None, None] * np.ones((1025, 1, 1))

    paths = []
    for i in range(2):
        cube = _make_spectral_cube(wcs, data.copy())
        path = tmp_path / f"cube_{i}.fits"
        cube.write(path, format="fits", overwrite=True)
        paths.append(path)

    averaged = ftsdata_average(paths, min_average_per_cpu=1)
    npt.assert_allclose(averaged.data, data, rtol=1e-6, atol=1e-8)
