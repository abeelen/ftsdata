"""OPD geometry detection, constructor dispatch and the OpdData alias."""

import numpy as np
import pytest
import astropy.units as u

from ftsdata import (
    DoubleSidedOpdData,
    FTSData,
    OneSidedOpdData,
    OpdData,
    SingleSidedOpdData,
)

from .models import gen_empty_ftsdata


@pytest.mark.parametrize(
    "itg_type, expected_cls",
    [
        ("ds", DoubleSidedOpdData),
        ("os", OneSidedOpdData),
        ("ss", SingleSidedOpdData),
        ("ss_low", SingleSidedOpdData),
    ],
)
def test_opd_geometry_dispatch(itg_type, expected_cls):
    """Check that the OPD geometry maps onto the right subclass."""
    data = gen_empty_ftsdata(n_itg=1024, n_pix=(1, 1), itg_type=itg_type)
    assert isinstance(data, expected_cls)


def test_from_array_dispatch():
    """OpdData.from_array dispatches on the OPD geometry."""
    data = np.zeros((11, 2))

    opd = np.arange(-5, 6) * 0.1 * u.mm
    assert isinstance(OpdData.from_array(opd, data), DoubleSidedOpdData)

    opd = np.arange(0, 11) * 0.1 * u.mm
    assert isinstance(OpdData.from_array(opd, data), OneSidedOpdData)

    opd = np.arange(-3, 8) * 0.1 * u.mm
    assert isinstance(OpdData.from_array(opd, data), SingleSidedOpdData)

    # An explicit subclass is always respected
    opd = np.arange(-5, 6) * 0.1 * u.mm
    assert isinstance(SingleSidedOpdData.from_array(opd, data), SingleSidedOpdData)


def test_reader_dispatch(gaussian_ss, tmp_path):
    """OpdData.read dispatches to the right subclass, FTSData.read does not."""
    filepath = str(tmp_path / "test_ss.fits")
    gaussian_ss.write(filepath, format="fits", overwrite=True)

    loaded = OpdData.read(filepath, format="fits")
    assert isinstance(loaded, SingleSidedOpdData)

    loaded = SingleSidedOpdData.read(filepath, format="fits")
    assert isinstance(loaded, SingleSidedOpdData)

    loaded = FTSData.read(filepath, format="fits")
    assert isinstance(loaded, FTSData)


def test_opddata_alias():
    """``OpdData`` is a backward-compatible alias of the base class."""
    assert OpdData is SingleSidedOpdData
    assert issubclass(DoubleSidedOpdData, OpdData)
    assert issubclass(OneSidedOpdData, OpdData)
    assert issubclass(FTSData, OpdData)
