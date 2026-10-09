#
# Copyright 2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Registries of functions that handle files by their type

A :class:`DataLoaders` selects the function by the MIME type of the file
and an optional predicate. A :class:`DataChecker` selects it by the name
of the instrument. The registries used by numina are in
:mod:`numina.core.config`.
"""

import warnings
import mimetypes


def is_fits_megara(pathname):
    """Check is any FITS"""
    if pathname.endswith(".fits"):
        return True
    else:
        return False


def is_fits_emir(pathname):
    """Check is any FITS"""
    if pathname.endswith(".fits"):
        return True
    else:
        return False


def is_json_structured(pathname):
    """Check is structured JSON"""
    import json

    # FIXME: I'm loading everything here
    with open(pathname) as fd:
        state = json.load(fd)

    if "type_fqn" in state:
        return True
    else:
        return False


class DataLoaders:
    """Registry of functions that handle a file, selected by its type.

    A function is registered with :meth:`register`, for a MIME type and,
    optionally, a predicate of the path. Calling the registry with a path
    calls the first registered function whose MIME type is the type of
    the file, as guessed from its name, and whose predicate is true.
    """

    def __init__(self):
        self._loaders = []

    def register(self, mtype, is_func=None, priority=20):
        """Decorator that registers a function.

        Parameters
        ----------
        mtype : str
            MIME type of the files handled, as 'image/fits'.
        is_func : callable, optional
            Predicate of the path, the function is used only if it
            returns True. By default, all the files of the type.
        priority : int, optional
            The functions with lower values are tried first.
        """

        if is_func is None:

            def is_func(p):
                return True  # noqa: E731

        def wrapper(func):
            self._loaders.append((priority, mtype, is_func, func))
            self._loaders.sort()
            return func

        return wrapper

    def dispatch(self, pathname):
        """Call the function that handles `pathname` and return its result.

        Raises
        ------
        TypeError
            If no function handles the file.
        """

        mmtype, enc = mimetypes.guess_type(pathname)
        # This is ordered by priority
        for priority, mtype, is_func, func in self._loaders:
            if (mmtype == mtype) and is_func(pathname):
                return func(pathname)
        else:
            raise TypeError(f"nothing handles {pathname}")

    def __call__(self, pathname):
        return self.dispatch(pathname)


class DataChecker:
    """Registry of functions that check objects, selected by instrument.

    A function is registered with :meth:`register` for the name of an
    instrument, and called as ``func(obj, astype=None, level=None)``.
    """

    def __init__(self):
        self._loaders = {}

    def __contains__(self, instrument_name):
        return instrument_name in self._loaders

    def register(self, instrument_name):
        """Decorator that registers the function of `instrument_name`"""

        def wrapper(func):
            self._loaders[instrument_name] = func
            return func

        return wrapper

    def dispatch(self, instrument, hdulist, astype=None, level=None):
        """Check `hdulist` with the function of `instrument`.

        If there is no function for the instrument, a warning is emitted
        and None is returned.
        """
        try:
            func = self._loaders[instrument]
        except KeyError:
            # No function registered
            warnings.warn(f"no function for {instrument}")
            return
        return func(hdulist, astype=astype, level=level)

    def __call__(self, instrument, hdulist, astype=None, level=None):
        return self.dispatch(instrument, hdulist, astype=astype, level=level)
