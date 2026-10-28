"""Blockwise/parallel execution engine."""

import numpy as np
import numpy.testing as npt
import pytest

from ftsdata import HAVE_DASK, apply_blockwise
from ftsdata import transforms


def _abs_rfft(block, mask_block, *, workers, axis, n_out):
    """A simple transform that keeps ``axis`` but changes its length."""
    if mask_block is not None:
        block = np.where(mask_block, 0, block)
    out = np.fft.rfft(block, n=2 * (n_out - 1), axis=axis)
    sl = [slice(None)] * out.ndim
    sl[axis] = slice(0, n_out)
    return np.ascontiguousarray(np.abs(out[tuple(sl)]))


def _transform(block, mask_block, *, workers):
    return _abs_rfft(block, mask_block, workers=workers, axis=0, n_out=21)


@pytest.fixture
def cube_and_mask():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(32, 64, 40))
    mask = rng.random(data.shape) < 0.1
    return data, mask


def _apply(func, data, *, mask=None, use_dask, small_cube_bytes):
    return apply_blockwise(
        func,
        data,
        0,
        21,
        mask=mask,
        out_dtype=np.float64,
        bytes_work_per_pix=data.dtype.itemsize * 8,
        small_cube_bytes=small_cube_bytes,
        ncpu=2,
        mem_available=1e12,
        use_dask=use_dask,
    )


def test_blockwise_matches_single_call(cube_and_mask):
    data, mask = cube_and_mask
    func = _transform

    blocked = _apply(func, data, mask=mask, use_dask=False, small_cube_bytes=0)
    single = _apply(func, data, mask=mask, use_dask=False, small_cube_bytes=np.inf)

    assert blocked.shape == (21, 64, 40)
    npt.assert_allclose(blocked, single, rtol=1e-10, atol=1e-10)


def test_blockwise_without_mask(cube_and_mask):
    data, _ = cube_and_mask
    func = _transform

    blocked = _apply(func, data, use_dask=False, small_cube_bytes=0)
    single = _apply(func, data, use_dask=False, small_cube_bytes=np.inf)
    npt.assert_allclose(blocked, single, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not HAVE_DASK, reason="dask is not installed")
def test_blockwise_dask_matches_threads(cube_and_mask):
    data, mask = cube_and_mask
    func = _transform

    threads = _apply(func, data, mask=mask, use_dask=False, small_cube_bytes=0)
    dask = _apply(func, data, mask=mask, use_dask=True, small_cube_bytes=0)
    npt.assert_allclose(threads, dask, rtol=1e-10, atol=1e-10)


def test_dask_requested_but_missing(monkeypatch, cube_and_mask):
    data, mask = cube_and_mask
    monkeypatch.setattr(transforms, "HAVE_DASK", False)
    func = _transform

    with pytest.raises(ImportError, match="dask"):
        _apply(func, data, mask=mask, use_dask=True, small_cube_bytes=0)


def test_output_does_not_fit_in_memory(cube_and_mask):
    data, mask = cube_and_mask
    func = _transform

    with pytest.raises(MemoryError):
        apply_blockwise(
            func,
            data,
            0,
            21,
            mask=mask,
            out_dtype=np.float64,
            bytes_work_per_pix=data.dtype.itemsize * 8,
            small_cube_bytes=0,
            ncpu=2,
            mem_available=1,
            use_dask=False,
        )
