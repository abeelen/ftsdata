"""Geometry, folding and basic container properties."""

import numpy as np
import numpy.testing as npt
import pytest
import astropy.units as u

from .models import gaussian_ss, gaussian_ss_low


def test_ftsdata_prop(empty_ds):
    data = empty_ds
    assert data._has_wcs_opd
    assert data._is_doublesided
    assert not data._is_onesided


def test_ftsdata_prop2(empty_os):
    data = empty_os
    assert data._has_wcs_opd
    assert not data._is_doublesided
    assert data._is_onesided


def test_extract_doublesided(gaussian_ds):
    data = gaussian_ds
    data_ds = data._extract_doublesided()

    assert np.all(data.data == data_ds.data)
    assert np.all(data.wcs.wcs.cdelt == data_ds.wcs.wcs.cdelt)
    assert np.all(data.wcs.wcs.crpix == data_ds.wcs.wcs.crpix)
    assert np.all(data.wcs.wcs.crval == data_ds.wcs.wcs.crval)


def test_extract_doublesided2(gaussian_ss):
    data = gaussian_ss
    data_ds = data._extract_doublesided()

    # 0 is put on the first quarter
    assert data_ds.shape[0] - 1 == (data.shape[0] - 1) // 2
    assert np.all(data_ds.data == data.data[: (data.shape[0] - 1) // 2 + 1])


def test_to_onesided(gaussian_ds):
    data = gaussian_ds
    data_os = data._to_onesided()
    assert data_os.shape[0] - 1 == (data.shape[0] - 1) // 2
    assert np.all(data_os.data == data.data[(data.shape[0] - 1) // 2 :])


def test_to_onesided_mask(gaussian_ds):
    data = gaussian_ds
    data_os_nomask = data._to_onesided()
    data.mask = np.zeros_like(data.data, dtype=bool)
    zpd_idx = int(data.opd_wcs.all_world2pix(0, 0)[0])
    data.mask[:zpd_idx] = True
    data_os = data._to_onesided()
    npt.assert_equal(data_os.mask, False)
    npt.assert_equal(data_os_nomask.data, data_os.data)


def test_to_onesided_alpha(gaussian_ds_1d):
    data = gaussian_ds_1d
    data_os = data._to_onesided()
    data_os_alpha = data._to_onesided(alpha=0.2)
    assert np.all(data_os_alpha.data == data_os.data)
    data_os_alpha = data._to_onesided(alpha=0.5)
    assert np.all(data_os_alpha.data == data_os.data)


@pytest.mark.parametrize("fixture", [gaussian_ss, gaussian_ss_low])
def test_singlesided_to_onesided(fixture):
    data = fixture()
    data_os = data._to_onesided()
    zpd_idx = data.wcs.sub([3]).world_to_pixel(0 * u.mm)
    assert data_os.shape[0] == np.abs(np.array([-1, data.shape[0]]) - zpd_idx).max().astype(int)
