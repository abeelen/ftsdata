"""Logging helpers for the ftsdata package.

This is a small, dependency-free replacement for :func:`autologging.logged`.
It preserves the attribute names used by the original decorator: classes get a
name-mangled ``__log`` attribute and functions get ``_log``.
"""

import logging
from inspect import isclass


def _generate_logger_name(obj):
    """Return the logger name used for *obj*."""
    if isclass(obj):
        return "{}.{}".format(obj.__module__, getattr(obj, "__qualname__", obj.__name__))
    return obj.__module__


def _add_logger_to(obj, logger_name=None):
    """Attach a :class:`logging.Logger` to *obj* and return *obj*."""
    logger = logging.getLogger(logger_name if logger_name else _generate_logger_name(obj))

    if isclass(obj):
        setattr(obj, f"_{obj.__name__}__log", logger)
    else:
        obj._log = logger

    return obj


def logged(obj):
    """Add a logger member to a decorated class or function.

    Parameters
    ----------
    obj : class, function or `logging.Logger`
        The object to decorate.  Passing a logger returns a decorator that uses
        that logger as the parent logger.

    Returns
    -------
    The decorated object, or a decorator when *obj* is a logger.
    """
    if isclass(obj) or callable(obj) and not isinstance(obj, logging.Logger):
        return _add_logger_to(obj)

    return lambda o: _add_logger_to(o, obj.name)
