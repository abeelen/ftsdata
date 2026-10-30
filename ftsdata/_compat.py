"""Compatibility shims for the legacy Python 3.8 / Astropy 5.2 stack.

The package keeps Python 3.8 (and therefore NumPy 1.24, SciPy 1.10 and
Astropy 5.2) as an explicit CONCERTO exception.  This module centralises the
workarounds needed for that old stack and is targeted for removal in
``ftsdata 0.2.0``; see ``docs/min_versions.md``.
"""

from __future__ import annotations

# Sentinel used to distinguish an omitted ``hits`` keyword from an explicit
# ``hits=None``.
_HITS_UNSET = object()


def resolve_hits(source, hits):
    """Resolve the ``hits`` value for a newly constructed container.

    Astropy < 6 rebuilds the right-hand arithmetic operand with
    ``cls(operand)`` before calling ``_arithmetic``.  ``NDData`` copies its
    standard attributes in that case but knows nothing about custom ones such
    as ``hits``, so inherit them from the source container.  The behaviour is
    harmless on newer astropy versions and keeps ``cls(other)`` copying
    consistent with the rest of the ``NDData`` attributes.

    Parameters
    ----------
    source : object or None
        The ``data`` argument passed to the container constructor, if any.
    hits : object
        The explicitly requested hit map, or the ``_HITS_UNSET`` sentinel.

    Returns
    -------
    hits : `~numpy.ndarray` or None
        The hit map to use for the new container.
    """
    if hits is not _HITS_UNSET:
        return hits
    if source is not None and hasattr(source, "hits"):
        return source.hits
    return None
