"""Blockwise execution and Fourier-transform kernels for ftsdata.

The transform axis is never split: every block receives the full transform
axis and returns ``n_out`` samples along it.  This module deliberately imports
no ftsdata container class, so it can be tested and reused on its own.
"""

from concurrent.futures import ThreadPoolExecutor

import astropy.constants as cst
import astropy.units as u
import numpy as np
from astropy.convolution.kernels import CustomKernel
from scipy.fft import ifft, irfft, rfft

from ._utils import available_memory, cpu_count

try:
    from numpy import trapezoid
except ImportError:  # very old Numpy only....
    from numpy import trapz as trapezoid

try:
    import dask
    import dask.array as da

    HAVE_DASK = True
except ImportError:
    HAVE_DASK = False

def get_ils_kernel(opdmax_mm, dfreq_GHz, n_points=1001, onesided_apodization=None):
    """Retrieve the ILS of a given FTS observation as an astropy.convolution.Kernel.

    Parameters
    ----------
    opdmax_mm: float
        the maximum opd of the observation in mm
    dfreq_GHz: float
        the requested frequency scale in GHz
    n_points : int, optionnal
        the number of points for the kernel (odd number), default 1001
    onesided_apodization : [function], optional
        apodization function to be used on the one sided interferograms, by default None

    Returns
    -------
    kernel : astropy.convolution.CustomKernel
        the kernel to be used for spectral convolution.
    """

    # What is the fourier transform of a rectangular function?
    # rect(x/a) = 1 si -a/2 < t < a/2, 0 sinon
    # FT(rect(x/a))(sigma) = np.abs(a) * sinc( f * a), w. sinc(x) = sin(pi*x)/(pi*x)
    # x axis is opd, sigma is wavenumber

    # ILS(wave) = (2 opdmax) * sinc( wave * (2 opdmax))
    # ILS(freq) = (2 opdmax) * sinc( freq / c * (2 opdmax))

    freq = np.linspace(-(n_points - 1) // 2, n_points // 2, n_points) * dfreq_GHz * u.GHz
    kernel = 2 * opdmax_mm * u.mm * np.sinc(((freq / cst.c) * (2 * opdmax_mm * u.mm)).decompose().value)
    kernel /= trapezoid(kernel, freq.to_value("GHz"))

    if onesided_apodization is not None:
        opd_mm = (np.fft.rfftfreq(n_points, d=dfreq_GHz * u.GHz) * cst.c).to_value("mm")
        window = np.zeros_like(opd_mm)
        mask = opd_mm <= opdmax_mm
        n_window = np.sum(mask)
        window[mask] = onesided_apodization(2 * n_window - 1)[n_window - 1 :]
        window = np.fft.fftshift(np.fft.irfft(window, n=n_points))

        kernel = np.convolve(kernel, window, mode="same")
        kernel /= trapezoid(kernel, freq.to_value("GHz"))

    return CustomKernel(kernel)


def _apply_blockwise_threads(func, data, axis, n_out, ax, rows, *, mask=None, out_dtype, ncpu):
    """Run ``func`` block by block with a :class:`ThreadPoolExecutor`.

    ``apply_blockwise`` has already computed the block geometry ``(ax, rows)``
    and the memory budget; this helper only schedules the work.
    """
    out_shape = list(data.shape)
    out_shape[axis] = n_out
    out = np.empty(out_shape, dtype=out_dtype)

    def work(i0):
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(i0, i0 + rows)
        sl = tuple(sl)
        out[sl] = func(np.asarray(data[sl]), None if mask is None else np.asarray(mask[sl]), workers=1)

    with ThreadPoolExecutor(max_workers=ncpu) as ex:
        list(ex.map(work, range(0, data.shape[ax], rows)))  # list() re-raises exceptions
    return out


def _apply_blockwise_dask(func, data, axis, n_out, ax, rows, *, mask=None, out_dtype, ncpu):
    """Run ``func`` block by block with dask.

    ``apply_blockwise`` has already computed the block geometry ``(ax, rows)``
    and the memory budget; this helper only builds and computes the dask graph.
    """
    # Only the pixel axis ``ax`` is chunked; ``axis`` stays whole in every block.
    chunks = [-1] * data.ndim
    chunks[ax] = rows

    ddata = da.from_array(np.asarray(data), chunks=tuple(chunks), name=False)

    # ``func`` keeps ``axis`` (it only changes its length to ``n_out``),
    # so the output chunks match the input ones except along ``axis``.
    out_chunks = list(ddata.chunks)
    out_chunks[axis] = (n_out,)

    if mask is not None:
        # Passing the mask as a dask array lets dask slice it per block,
        # instead of handing the full mask to every task.
        dmask = da.from_array(np.asarray(mask), chunks=tuple(chunks), name=False)

        def block_func(block, mask_block, **kwargs):
            return func(block, mask_block, workers=1)

        graph = da.map_blocks(
            block_func,
            ddata,
            dmask,
            dtype=out_dtype,
            chunks=tuple(out_chunks),
            meta=np.array((), dtype=out_dtype),
        )
    else:

        def block_func(block, **kwargs):
            return func(block, None, workers=1)

        graph = da.map_blocks(
            block_func,
            ddata,
            dtype=out_dtype,
            chunks=tuple(out_chunks),
            meta=np.array((), dtype=out_dtype),
        )

    with dask.config.set(scheduler="threads", num_workers=ncpu):
        return graph.compute()


def apply_blockwise(
    func,
    data,
    axis,
    n_out,
    *,
    mask=None,
    out_dtype,
    bytes_work_per_pix,
    mem_safety=0.8,
    small_cube_bytes=1e7,
    ncpu=None,
    mem_available=None,
    use_dask=None,
):
    """Apply ``func`` to a cube in parallel, over blocks of pixels.

    The transform ``axis`` is never sliced: every task receives the full
    extent of ``data`` along ``axis`` and returns ``n_out`` samples along
    that axis. The remaining (pixel) axes are split into contiguous blocks,
    one per thread of a :class:`concurrent.futures.ThreadPoolExecutor`,
    or one per dask worker if ``use_dask`` is True.

    Parameters
    ----------
    func : callable
        Function applied to each block, called as
        ``func(block, mask_block, workers=...)``. ``block`` is a
        :class:`numpy.ndarray` with the pixel axes restricted to the block
        and the transform axis complete. ``mask_block`` is the matching
        slice of ``mask``, or ``None`` if ``mask`` is ``None``. It must
        return an array with the same shape as ``block``, except along
        ``axis`` where it must have ``n_out`` samples.
    data : array_like
        Input cube. It is only read, never modified in place.
    axis : int
        Index of the transform axis, kept whole in every block.
    n_out : int
        Size of the output along ``axis``.
    mask : array_like, optional
        Boolean mask with the same shape as ``data``, passed to ``func``
        block by block. Default is ``None``.
    out_dtype : numpy.dtype
        dtype of the allocated output array.
    bytes_work_per_pix : float
        Estimated peak memory, in bytes, used by ``func`` per output pixel.
        Together with the available memory and the number of threads, it
        sets the size of a block.
    mem_safety : float, optional
        Fraction of the currently available memory that the output plus the
        working blocks may use. Default is 0.8.
    small_cube_bytes : float, optional
        If the total estimated work is below this value, or if ``data`` has
        no pixel axis, ``func`` is called once on the whole cube with
        ``workers=ncpu`` instead of block by block. Default is 1e7.
    ncpu : int, optional
        Number of worker threads. Default is ``None``, which uses
        :func:`ftsdata._utils.cpu_count`.
    mem_available : float, optional
        Available memory, in bytes. Default is ``None``, which uses
        :func:`ftsdata._utils.available_memory`.
    use_dask : bool, optional
        If True, use dask to process the cube in parallel.
        If None (default), use dask if available.
        If False, use ThreadPoolExecutor.

    Returns
    -------
    out : numpy.ndarray
        Output array with the same shape as ``data``, except along ``axis``
        where it has ``n_out`` samples.

    Raises
    ------
    MemoryError
        If the output alone does not fit in the available memory, or if one
        row of pixels does not fit in the per-thread memory budget.

    Notes
    -----
    When ``ncpu`` and ``mem_available`` are not given, they are read with
    :func:`ftsdata._utils.cpu_count` and
    :func:`ftsdata._utils.available_memory`.

    In the block path, ``func`` is called with ``workers=1`` because
    parallelism already comes from the thread pool/dask; the single-call path
    uses ``workers=ncpu``. ``func`` may ignore that keyword if it does not
    parallelise internally.
    """
    other_axes = [ax for ax in range(data.ndim) if ax != axis]
    npix = int(np.prod([data.shape[ax] for ax in other_axes])) if other_axes else 1

    # Memory budget
    ncpu = cpu_count() if ncpu is None else ncpu
    mem_available = available_memory() if mem_available is None else mem_available

    mem_output = npix * n_out * np.dtype(out_dtype).itemsize
    mem_free = mem_safety * mem_available - mem_output
    if mem_free <= 0:
        raise MemoryError("Output does not fit in memory")

    work_total = npix * bytes_work_per_pix
    if not other_axes or work_total < small_cube_bytes:
        # 1D or small enough: single direct call
        return func(np.asarray(data), None if mask is None else np.asarray(mask), workers=ncpu)

    pix_per_block = int((mem_free / ncpu) / bytes_work_per_pix)

    ax = other_axes[int(np.argmax([data.shape[i] for i in other_axes]))]
    pix_per_row = int(np.prod([data.shape[i] for i in other_axes if i != ax]))
    if pix_per_block < pix_per_row:
        raise MemoryError(
            "One row of pixels exceeds the per-task budget: chunk a second spatial axis too, or reduce the output size."
        )
    rows = max(1, pix_per_block // pix_per_row)

    # Resolve the requested executor
    if use_dask is None:
        use_dask = HAVE_DASK
    if use_dask and not HAVE_DASK:
        raise ImportError("use_dask=True but dask is not installed")

    executor = _apply_blockwise_dask if use_dask else _apply_blockwise_threads
    return executor(
        func,
        data,
        axis,
        n_out,
        ax,
        rows,
        mask=mask,
        out_dtype=out_dtype,
        ncpu=ncpu,
    )



def _to_itg_block(block, mask, *, workers, axis, n, phase_norm, weights):
    wshape = [1] * block.ndim
    wshape[axis] = -1

    # Check GIL problems
    itg = rfft(block * weights.reshape(wshape), n=n, axis=axis, workers=workers)
    # itg = np.fft.rfft(block * weights.reshape(wshape), n=n, axis=axis)  # no workers= argument

    # Apply the precomputed phase ramp with the norm folded in, see to_itg().
    # `phase_norm` is already in the dtype returned by rfft for this input.
    itg *= phase_norm.reshape(wshape)

    return np.ascontiguousarray(itg.real)


def get_itg_binning_correction(freq, delta_opd):
    """Compute the itg binning correction to spectra

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        the frequency axis of the spectrum
    delta_opd : `~astropy.units.Quantity`
        the delta opd of the original interferogram

    Returns
    -------
    correction : `~numpy.ndarray` or float
        the correction to apply to spectra, same shape as freq
    """
    # Normalised-sinc argument: ν [GHz] × (Δopd / c) [1/GHz]
    sinc_arg = freq * (delta_opd / cst.c)
    correction = np.sinc(sinc_arg.decompose().value)  # shape (n_freq,)

    return correction


def _to_spectra_doublesided_block(block, mask, *, workers, axis, output_shape, weights, peak, real=True):
    """block: double-sided interferograms, `axis` = OPD axis (whole).
    Returns output_shape // 2 + 1 frequency samples along `axis`."""
    wshape = [1] * block.ndim
    wshape[axis] = -1

    # Apodization x norm_factor in a single new array, then zero the masked
    # samples in place: one full-size temporary instead of two.
    x = block * weights.reshape(wshape)
    if mask is not None:
        np.copyto(x, 0, where=mask)  # masked -> 0, in place

    # Pad PRIOR to the FFT, otherwise the transform is not correct when n_pad
    # increases the output sampling.
    naxis_opd = x.shape[axis]
    if output_shape != naxis_opd:
        pad_size = output_shape - naxis_opd
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (pad_size - pad_size // 2, pad_size // 2)
        x = np.pad(x, pad_width=pad_width, mode="constant", constant_values=0)

    # Move the ZPD (zero frequency) to the start of the OPD axis, as numpy's fft
    # expects it at index 0.  This must happen *after* padding: ifftshift centres
    # on the padded length, which is what puts the ZPD exactly at index 0.
    x = np.fft.ifftshift(x, axes=axis)

    spectra = ifft(x, n=output_shape, axis=axis, workers=workers)

    # Positive frequencies only; .copy() so the big ifft buffer is freed on return
    sl = [slice(None)] * spectra.ndim
    sl[axis] = slice(0, output_shape // 2 + 1)
    if peak or real:
        # A symmetric interferogram has a real spectrum; drop the residual
        # imaginary part introduced by numerical noise / masking.
        return spectra[tuple(sl)].real.copy()

    return spectra[tuple(sl)].copy()


def _to_spectra_onesided_block(block, mask, *, workers, axis, output_shape, weights):
    """block: one-sided interferograms, `axis` = OPD axis (whole).
    Returns output_shape // 2 + 1 frequency samples along `axis`."""
    wshape = [1] * block.ndim
    wshape[axis] = -1

    # Apodization x norm_factor in a single new array, then zero the masked
    # samples in place: one full-size temporary instead of two.
    x = block * weights.reshape(wshape)
    if mask is not None:
        np.copyto(x, 0, where=mask)  # masked -> 0, in place

    spectra = irfft(x, n=output_shape, axis=axis, workers=workers)
    # spectra = np.fft.irfft(x, n=output_shape, axis=axis)

    # Positive frequencies only; .copy() so the big irfft buffer is freed on return
    sl = [slice(None)] * block.ndim
    sl[axis] = slice(0, output_shape // 2 + 1)
    return spectra[tuple(sl)].copy()
