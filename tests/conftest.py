"""Shared fixtures for the ftsdata test-suite."""

import pytest

from .models import (
    gaussian_ds,
    gaussian_ds_1d,
    gaussian_ds_2d,
    gaussian_ds_noise,
    gaussian_ds_shift,
    gaussian_os,
    gaussian_os_noise,
    gaussian_ss,
    gaussian_ss_low,
    gaussian_ss_shift,
    gen_empty_ftsdata,
    rect_ds,
    rect_ds_shift,
    rect_os,
    rect_ss,
    rect_ss_shift,
)


@pytest.fixture(name="empty_ds")
def empty_ds_fixture():
    return gen_empty_ftsdata(n_itg=1024, n_pix=(1, 1), itg_type="ds")


@pytest.fixture(name="empty_ds_2d")
def empty_ds_2d_fixture():
    return gen_empty_ftsdata(n_itg=1024, n_pix=1, itg_type="ds")


@pytest.fixture(name="empty_ds_1d")
def empty_ds_1d_fixture():
    return gen_empty_ftsdata(n_itg=1024, n_pix=None, itg_type="ds")


@pytest.fixture(name="empty_os")
def empty_os_fixture():
    return gen_empty_ftsdata(n_itg=1024, n_pix=(1, 1), itg_type="os")


@pytest.fixture(name="empty_ss")
def empty_ss_fixture():
    return gen_empty_ftsdata(n_itg=1024, n_pix=(1, 1), itg_type="ss")


# The factories below are cheap wrappers around ``models.fixture_itg``.  They
# are exposed as fixtures so each test gets a fresh, independent cube.


@pytest.fixture(name="gaussian_ds")
def fixture_gaussian_ds():
    return gaussian_ds()


@pytest.fixture(name="gaussian_ds_2d")
def fixture_gaussian_ds_2d():
    return gaussian_ds_2d()


@pytest.fixture(name="gaussian_ds_1d")
def fixture_gaussian_ds_1d():
    return gaussian_ds_1d()


@pytest.fixture(name="gaussian_ds_noise")
def fixture_gaussian_ds_noise():
    return gaussian_ds_noise()


@pytest.fixture(name="gaussian_os_noise")
def fixture_gaussian_os_noise():
    return gaussian_os_noise()


@pytest.fixture(name="rect_ds")
def fixture_rect_ds():
    return rect_ds()


@pytest.fixture(name="gaussian_os")
def fixture_gaussian_os():
    return gaussian_os()


@pytest.fixture(name="rect_os")
def fixture_rect_os():
    return rect_os()


@pytest.fixture(name="gaussian_ss")
def fixture_gaussian_ss():
    return gaussian_ss()


@pytest.fixture(name="gaussian_ss_low")
def fixture_gaussian_ss_low():
    return gaussian_ss_low()


@pytest.fixture(name="rect_ss")
def fixture_rect_ss():
    return rect_ss()


@pytest.fixture(name="gaussian_ds_shift")
def fixture_gaussian_ds_shift():
    return gaussian_ds_shift()


@pytest.fixture(name="gaussian_ss_shift")
def fixture_gaussian_ss_shift():
    return gaussian_ss_shift()


@pytest.fixture(name="rect_ds_shift")
def fixture_rect_ds_shift():
    return rect_ds_shift()


@pytest.fixture(name="rect_ss_shift")
def fixture_rect_ss_shift():
    return rect_ss_shift()
