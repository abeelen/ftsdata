"""Preview plots and weighted co-addition for ftsdata cubes."""

import logging
from functools import partial
from itertools import chain, product
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import InverseVariance
from astropy.stats import mad_std

from ._utils import available_memory, cpu_count, merge_metas, safe_map
from .core import FTSData, SpecData

logger = logging.getLogger("FTSData")

def plot_cube_preview(spectral_cubes, freq_range_GHz=[100, 400], ncols=8, spectrum_coord=(0, 0), **kwargs):
    if isinstance(spectral_cubes, FTSData):
        spectral_cubes = [spectral_cubes]

    ncubes = len(spectral_cubes)

    # Half of the height is reserved for the spectra
    nrows = 2 * ncubes

    import matplotlib.pyplot as plt

    fig = plt.figure()
    gs = fig.add_gridspec(ncols=ncols, nrows=nrows)

    ax_spec = fig.add_subplot(gs[nrows // 2 :, :])
    axes = [fig.add_subplot(gs[i, j]) for i, j in product(range(ncubes), range(ncols))]
    axes = np.array(axes).reshape(ncubes, ncols)

    freqs_GHz = np.linspace(*freq_range_GHz, ncols)
    ax_spec.set_xlim(*freq_range_GHz)

    for _axes, spec in zip(axes, spectral_cubes):
        x, y = np.round(spec.wcs.celestial.all_world2pix(*spectrum_coord, 0)).astype(int)

        ax_spec.plot(spec.spectral_axis.to("GHz"), spec.data[:, y, x])

        for ax, freq_GHz in zip(_axes, freqs_GHz):
            idx = spec.wcs.spectral.world_to_pixel(freq_GHz * u.GHz).astype(int)
            ax.imshow(spec.data[idx], **kwargs)
            ax.set_aspect("equal")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

    for ax, freq_GHz in zip(axes[0], freqs_GHz):
        ax.set_title("{:3.1f} GHz".format(freq_GHz))

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def _ftsdata_generator(ftsdata):

    for item in ftsdata:
        if isinstance(item, FTSData):
            yield item
        elif isinstance(item, (Path, str)):
            yield FTSData.read(item)
        else:
            raise ValueError(f"Do not know how to handle {item}")


def _get_ftsdata(item, nan_threshold=0.3, zpd_std=False, clip_weight=True, normalize=True, wdata=True):
    wcs = item.wcs
    data = item.data
    mask = item.mask
    hit = item.hits
    weight = item.weights

    if normalize:
        snr = data * np.sqrt(weight)
        factor = mad_std(snr[~mask])
        weight /= factor**2

    nan_fraction = np.isnan(data).mean(axis=tuple(range(1, data.ndim)))
    # normalize nan_fraction
    nan_fraction = (nan_fraction - nan_fraction.min()) / (nan_fraction.max() - nan_fraction.min())

    bad_data = nan_fraction > nan_threshold
    weight[bad_data] = 0
    hit[bad_data] = 0

    if zpd_std:
        zpd_idx = int(wcs.wcs.crpix[-1] - 1)
        std = np.nanstd(data[zpd_idx])
        weight *= 1 / (std * std)

    if clip_weight:
        mean_weight = np.nanmean(weight[~mask])
        std_weight = np.nanstd(weight[~mask])
        weight[weight > (mean_weight + 4 * std_weight)] = 0

    data[mask] = 0.0
    weight[mask] = 0.0
    hit[mask] = 0

    if wdata:
        data *= weight

    return wcs, data, weight, mask, hit


def _ftsdata_average_worker(ftsdatas, zpd_std=False, clip_weight=True, normalize=True):
    items = _ftsdata_generator(ftsdatas)

    _get_ftsdata_kwargs = dict(normalize=normalize, zpd_std=zpd_std, clip_weight=clip_weight, wdata=True)

    item = next(items)
    meta = [item.meta]
    wcs, data, weight, mask, hit = _get_ftsdata(item, **_get_ftsdata_kwargs)

    for item in items:
        meta.append(item.meta)
        _wcs, _data, _weight, _mask, _hit = _get_ftsdata(item, **_get_ftsdata_kwargs)

        if wcs.wcs != _wcs.wcs:
            logger.warning("WCS not equal, skipping")
            continue

        data += _data
        weight += _weight
        hit += _hit

    return wcs, data, weight, hit, meta


def ftsdata_average(ftsdatas, normalize=True, zpd_std=False, clip_weight=True, min_average_per_cpu=5):
    """Return weighted average of severak FTSData, using inverse variance as the weights

    Parameters
    ----------
    datas: list of class:`kidsdata.ftsdata.FTSData`
        the list of FTSData object
    normalize : bool, TBD
        normalize the uncertainty such that the snr std is 1, default False
    min_average_per_cpu: int
        minimum number of cube to be average per cpu

    Returns
    -------
    data : class:`kidsdata.ftsdata.FTSData`
        the resulting combined filtered FTSData object

    Notes
    -----
        All ftsdatas are in memory here, so we can use numpy to speed up the process

    """
    ref = ftsdatas[0]
    hdr = ref.meta if isinstance(ref, (SpecData, FTSData)) else fits.getheader(ref, "DATA")
    shape = tuple(hdr.get(f"NAXIS{i}") for i in range(hdr["NAXIS"], 0, -1))

    cube_memory = np.prod(shape) * np.dtype("float32").itemsize * 3
    memory_per_worker = cube_memory * 3

    free_mem = available_memory() / 1.20

    n_cpu = min(
        [
            int(free_mem / memory_per_worker),
            int(free_mem / (cube_memory + 2)),
            int(len(ftsdatas) / min_average_per_cpu),
            cpu_count(),
        ]
    )

    if n_cpu <= 0:
        raise MemoryError("Not enough memory to average the cubes")

    if n_cpu < cpu_count():
        logger.warning("Using %s CPUs instead", n_cpu)

    # Map step :
    _this = partial(_ftsdata_average_worker, normalize=normalize, zpd_std=zpd_std, clip_weight=clip_weight)
    results = safe_map(_this, np.array_split(ftsdatas, n_cpu), processes=n_cpu)

    # Reduce step :
    item = results[0]
    wcs = item[0]
    data = [item[1]]
    weight = [item[2]]
    hit = [item[3]]
    meta = [item[4]]

    for _wcs, _data, _weight, _hit, _meta in results[1:]:
        if wcs.wcs != _wcs.wcs:
            logger.warning("WCS not equal, skipping")
            continue
        data.append(_data)
        weight.append(_weight)
        hit.append(_hit)
        meta.append(_meta)

    data = np.asarray(data)
    weight = np.asarray(weight)
    hit = np.asarray(hit)
    meta = merge_metas(chain.from_iterable(meta))
    del results

    hit = hit.sum(0)
    weight = weight.sum(0)
    data = data.sum(0) / weight
    mask = np.isnan(data)

    if normalize:
        snr = data * np.sqrt(weight)
        factor = mad_std(snr[~mask])
        weight /= factor**2
    # TODO: meta data should be updated
    from astropy.nddata import InverseVariance

    output = FTSData(data=data, uncertainty=InverseVariance(weight), wcs=wcs, hits=hit, mask=mask, meta=meta)

    # if normalize:
    #     output.normalize_uncertainty()

    return output
