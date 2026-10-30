"""Arithmetic propagation of the ``hits`` attribute.

``NDArithmeticMixin`` propagates data, uncertainty, mask, WCS and metadata but
knows nothing about ``hits``.  ``HitsMixinClass`` adds that propagation; these
tests pin down the resulting behaviour, including the interaction with
uncertainty propagation and with collapse operations.
"""

from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest

from .models import gaussian_ds, gaussian_ds_2d


@pytest.mark.parametrize(
    "operation",
    ["add", "subtract", "multiply", "divide"],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
def test_binary_arithmetic_combines_hits(gaussian_ds_2d, operation):
    """Binary operations return the sum of the hit counts by default."""
    first = gaussian_ds_2d
    # A deepcopy keeps the custom ``hits`` attribute, unlike a plain
    # ``cls(other)`` conversion on older astropy versions.
    second = deepcopy(first)
    second.hits[...] = 3

    result = getattr(first, operation)(second)

    assert isinstance(result, first.__class__)
    npt.assert_array_equal(result.hits, first.hits + second.hits)
    assert np.issubdtype(result.hits.dtype, np.integer)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_binary_arithmetic_propagates_uncertainties(gaussian_ds_2d):
    """Adding hits must not interfere with the parent uncertainty handling."""
    first = gaussian_ds_2d
    second = gaussian_ds_2d

    result = first.add(second)

    # The fixture uses independent StdDevUncertainty ones, so adding gives
    # sqrt(1**2 + 1**2).  This regresses a bug where the extra ``operand2``
    # positional argument was interpreted as ``propagate_uncertainties=None``.
    npt.assert_allclose(result.uncertainty.array, np.sqrt(2))

    result = first.add(second, propagate_uncertainties=False)
    assert isinstance(result.uncertainty, first.uncertainty.__class__)


def test_arithmetic_with_scalar_or_array_keeps_hits(gaussian_ds_2d):
    """An operand without hits does not alter the hit map of the data."""
    data = gaussian_ds_2d
    expected = data.hits.copy()

    for operand in (2, np.ones_like(data.data)):
        result = data.add(operand)
        npt.assert_array_equal(result.hits, expected)


def test_arithmetic_only_one_operand_has_hits(gaussian_ds_2d):
    """Hits are taken from whichever operand carries them."""
    with_hits = gaussian_ds_2d
    without_hits = deepcopy(with_hits)
    without_hits.hits = None

    npt.assert_array_equal(with_hits.add(without_hits).hits, with_hits.hits)
    npt.assert_array_equal(without_hits.add(with_hits).hits, with_hits.hits)

    both_without_hits = deepcopy(with_hits)
    both_without_hits.hits = None
    assert both_without_hits.add(without_hits).hits is None


def test_handle_hits_first_found(gaussian_ds_2d):
    """``first_found`` behaves like the equivalent astropy mask policy."""
    first = gaussian_ds_2d
    second = deepcopy(first)
    second.hits[...] = 3

    npt.assert_array_equal(first.add(second, handle_hits="first_found").hits, first.hits)
    npt.assert_array_equal(first.add(second, handle_hits="ff").hits, first.hits)

    first.hits = None
    npt.assert_array_equal(first.add(second, handle_hits="first_found").hits, second.hits)
    assert first.add(second, handle_hits=None).hits is None


def test_handle_hits_callable(gaussian_ds_2d):
    """A custom callable receives both hit maps."""
    first = gaussian_ds_2d
    second = deepcopy(first)
    second.hits[...] = 3

    result = first.add(second, handle_hits=np.maximum)
    npt.assert_array_equal(result.hits, np.maximum(first.hits, second.hits))


def test_handle_hits_invalid(gaussian_ds_2d):
    """Unknown policies fail instead of silently dropping hits."""
    with pytest.raises(ValueError, match="handle_hits"):
        gaussian_ds_2d.add(deepcopy(gaussian_ds_2d), handle_hits="unknown")


def test_constructor_inherits_hits(gaussian_ds_2d):
    """Building a container from another one keeps its custom hit map."""
    original = gaussian_ds_2d
    original.hits[...] = 2

    clone = type(original)(original)
    npt.assert_array_equal(clone.hits, original.hits)

    stripped = type(original)(original, hits=None)
    assert stripped.hits is None


@pytest.mark.parametrize("operation", ["sum", "mean", "min", "max"])
def test_collapse_operations_keep_hits_shape(gaussian_ds_2d, operation):
    """Collapse operations work and reduce hits together with the data."""
    data = gaussian_ds_2d
    if not hasattr(data, operation):
        pytest.skip("collapse arithmetic requires astropy >= 6")

    result = getattr(data, operation)()
    assert np.shape(result.hits) == np.shape(result.data)

    result = getattr(data, operation)(axis=1)
    assert np.shape(result.hits) == np.shape(result.data)


def test_specdata_slicing_hits_uses_spatial_axes(gaussian_ds):
    """Spectral-cube hits are spatial, so the spectral axis is removed."""
    spectrum = gaussian_ds.to_spectra()
    assert spectrum.hits.shape == spectrum.data.shape[1:]

    sliced = spectrum[slice(0, 3), 0, 0]

    npt.assert_array_equal(sliced.hits, spectrum.hits[0, 0])
