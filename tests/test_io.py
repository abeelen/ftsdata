"""FITS/HDF5 round-trips and HDF5 dataset layout."""

import numpy as np
import numpy.testing as npt
import pytest

from ftsdata import FTSData, OpdData, SpecData

from .models import gaussian_ds


@pytest.mark.parametrize("fixture", [gaussian_ds])
def test_mip(fixture):
    data = fixture(n_pix=2**3)
    mip_data = fixture(n_pix=2**2)
    assert np.all(data.mip(axes=(1, 2), level=2) == mip_data.data)

    with pytest.raises(ValueError):
        data.mip(axes=(1, 2), level=3)


@pytest.mark.parametrize("fixture", [gaussian_ds])
def test_to_dset(fixture):
    data = fixture(n_pix=2**4)

    dsets, attrs = data.to_dset(planes=["XY", "XYZ"])
    for key in ["DATA", "HITS", "UNCERT", "PERCENTILE_RANKS", "Statistics"]:
        assert key in dsets
    assert np.all(dsets["DATA"] == data.data.astype(np.float32))
    assert np.all(dsets["HITS"] == data.hits)
    assert np.all(dsets["UNCERT"] == data.uncertainty.array)

    for plane in dsets["Statistics"]:
        assert plane in ["XY", "XYZ"]
        for stat in ["MAX", "MIN", "SUM", "SUM_SQ", "NAN_COUNT", "PERCENTILES"]:
            assert stat in dsets["Statistics"][plane]

    assert "0" in attrs
    assert attrs["0"]["WCSAXES"] == 3
    assert attrs["0"]["NAXIS1"] == 2**4
    assert attrs["0"]["NAXIS2"] == 2**4
    assert "UNCERT" in attrs
    assert attrs["UNCERT"]["UTYPE"] == "StdDevUncertainty"

    dsets, attrs = data.to_dset(mip_levels=[2, 4])
    assert "MipsMaps" in dsets
    assert "DATA" in dsets["MipsMaps"]
    for mip_level in [2, 4]:
        key = f"DATA_XY_{mip_level}"
        assert key in dsets["MipsMaps"]["DATA"]
        mip_data = fixture(n_pix=2 ** (4 - mip_level // 2))
        assert np.all(dsets["MipsMaps"]["DATA"][key] == mip_data.data.astype(np.float32))


def test_opddata_fits_roundtrip(gaussian_ds, tmp_path):
    """Test OpdData FITS write/read roundtrip."""
    filepath = str(tmp_path / "test_opd.fits")
    gaussian_ds.write(filepath, format="fits", overwrite=True)
    loaded = OpdData.read(filepath, format="fits")
    assert isinstance(loaded, OpdData)
    npt.assert_array_equal(loaded.data, gaussian_ds.data)
    assert loaded.wcs.to_header() == gaussian_ds.wcs.to_header()


def test_specdata_fits_roundtrip(gaussian_ds, tmp_path):
    """Test SpecData FITS write/read roundtrip."""
    spec = gaussian_ds.to_spectra()
    filepath = str(tmp_path / "test_spec.fits")
    spec.write(filepath, format="fits", overwrite=True)
    loaded = SpecData.read(filepath, format="fits")
    assert isinstance(loaded, SpecData)
    npt.assert_array_almost_equal(loaded.data, spec.data)
    npt.assert_allclose(loaded.spectral_axis.to_value("Hz"), spec.spectral_axis.to_value("Hz"), rtol=1e-10)


def test_ftsdata_fits_roundtrip(gaussian_ds, tmp_path):
    """Test FTSData (retro-compat class) FITS write/read roundtrip."""
    opd = FTSData(
        gaussian_ds.data,
        wcs=gaussian_ds.wcs,
        hits=gaussian_ds.hits,
        mask=gaussian_ds.mask,
        uncertainty=gaussian_ds.uncertainty,
    )
    filepath = str(tmp_path / "test_fts.fits")
    opd.write(filepath, format="fits", overwrite=True)
    loaded = FTSData.read(filepath, format="fits")
    assert isinstance(loaded, FTSData)
    npt.assert_array_equal(loaded.data, opd.data)


def test_opddata_hdf5_roundtrip(gaussian_ds, tmp_path):
    """Test OpdData HDF5 write/read roundtrip."""
    filepath = str(tmp_path / "test_opd.hdf5")
    gaussian_ds.write(filepath, format="hdf5")
    loaded = OpdData.read(filepath, format="hdf5")
    assert isinstance(loaded, OpdData)
    npt.assert_array_almost_equal(loaded.data, gaussian_ds.data)
    assert loaded.wcs.to_header() == gaussian_ds.wcs.to_header()


def test_specdata_hdf5_roundtrip(gaussian_ds, tmp_path):
    """Test SpecData HDF5 write/read roundtrip."""
    spec = gaussian_ds.to_spectra()
    filepath = str(tmp_path / "test_spec.hdf5")
    spec.write(filepath, format="hdf5")
    loaded = SpecData.read(filepath, format="hdf5")
    assert isinstance(loaded, SpecData)
    npt.assert_array_almost_equal(loaded.data, spec.data)
    npt.assert_allclose(loaded.spectral_axis.to_value("Hz"), spec.spectral_axis.to_value("Hz"), rtol=1e-10)
