import contextlib
import os


@contextlib.contextmanager
def working_directory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    """

    # http://code.activestate.com/recipes/576620-changedirectory-context-manager/

    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextlib.contextmanager
def ignored(*exceptions):

    # by Raymond Hettinger
    # https://www.youtube.com/watch?v=OSGv2VnC0go&t=1215s

    try:
        yield
    except exceptions:
        pass


@contextlib.contextmanager
def environ(**keys):
    """Context manager that recors environment"""
    old_vals = {key: os.environ.get(key, None) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, val in old_vals.items():
            if val is None:
                del os.environ[key]
            else:
                os.environ[key] = val
