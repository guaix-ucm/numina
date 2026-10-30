import mimetypes

import heapq

_loaders = []


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
    """Registry of functions that load files, selected by the MIME type of the file."""

    def __init__(self):
        self._loaders = []

    def register(self, mtype, is_func=None, priority=20):
        """Return a decorator that registers a loader of files of MIME type `mtype`.

        The loader is used only if ``is_func(pathname)`` is True, for all the
        files of the type if `is_func` is None. The loaders are stored in a
        heap by `priority`.
        """

        if is_func is None:

            def is_func(p):
                return True  # noqa: E731

        def wrapper(func):
            heapq.heappush(self._loaders, (priority, mtype, is_func, func))
            return func

        return wrapper

    def dispatch(self, pathname):
        """Load `pathname` with the first registered loader that accepts it.

        Raises
        ------
        TypeError
            If no loader accepts the file.
        """

        mmtype, enc = mimetypes.guess_type(pathname)
        # This is ordered by priority
        for priority, mtype, is_func, func in self._loaders:
            if (mmtype == mtype) and is_func(pathname):
                return func(pathname)
        else:
            raise TypeError(f"nothing loads {pathname}")

    def __call__(self, pathname):
        """Load `pathname`, as ``dispatch``."""
        return self.dispatch(pathname)


if __name__ == "__main__":

    load = DataLoaders()

    @load.register("image/fits", priority=20)
    def load_fits_0(pathname):
        import astropy.io.fits as fits

        return fits.open(pathname)

    @load.register("image/fits", is_fits_megara, priority=19)
    def load_fits_1(pathname):
        import astropy.io.fits as fits

        return fits.open(pathname)

    @load.register("image/fits", is_fits_emir, priority=5)
    def load_fits_2(pathname):
        import astropy.io.fits as fits

        return fits.open(pathname)

    @load.register("application/json", priority=20)
    def load_json(pathname):
        import json

        with open(pathname) as fd:
            return json.load(fd)

    @load.register("application/json", is_json_structured, priority=5)
    def load_json(pathname):  # noqa: F811
        import json
        from numina.util.objimport import import_object

        with open(pathname) as fd:
            data = json.load(fd)
        type_fqn = data["type_fqn"]
        cls = import_object(type_fqn)
        obj = cls.__new__(cls)
        obj.__setstate__(data)
        return obj

    fname = "/home/spr/devel/guaix/megaradrp/problem_marisa/data/master_traces.json"
    print(is_json_structured(fname))
    value = load.dispatch(fname)
    print(value)
