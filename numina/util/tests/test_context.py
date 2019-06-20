

import os
from ..context import environ


def test_environ_context1():
    """Remove env var if it was not set"""

    cache_dir = '/cache/dir'

    with environ(OOO_CACHE_HOME=cache_dir):
        assert os.environ['OOO_CACHE_HOME'] == cache_dir

    is_in = 'OOO_CACHE_HOME' in os.environ

    assert is_in == False


def test_environ_context2():
    """Reset env var if it was set"""

    cache_dir1 = '/cache/dir/1'
    cache_dir2 = '/cache/dir/2'

    os.environ['OOO_CACHE_HOME'] = cache_dir1

    with environ(OOO_CACHE_HOME=cache_dir2):
        assert os.environ['OOO_CACHE_HOME'] == cache_dir2

    assert os.environ['OOO_CACHE_HOME'] == cache_dir1


def test_environ_context3():
    """Reset multiple variables"""

    cache_dir1a = '/cache/dir/1a'
    cache_dir2a = '/cache/dir/2a'
    cache_dir3a = '/cache/dir/3a'
    cache_dir1b = '/cache/dir/1b'
    cache_dir2b = '/cache/dir/2b'

    os.environ['OOO_CACHE_HOME1'] = cache_dir1a
    os.environ['OOO_CACHE_HOME2'] = cache_dir2a
    os.environ['OOO_CACHE_HOME3'] = cache_dir3a

    with environ(OOO_CACHE_HOME1=cache_dir1b,
                 OOO_CACHE_HOME2=cache_dir2b):
        assert os.environ['OOO_CACHE_HOME1'] == cache_dir1b
        assert os.environ['OOO_CACHE_HOME2'] == cache_dir2b
        assert os.environ['OOO_CACHE_HOME3'] == cache_dir3a

    assert os.environ['OOO_CACHE_HOME1'] == cache_dir1a
    assert os.environ['OOO_CACHE_HOME2'] == cache_dir2a
    assert os.environ['OOO_CACHE_HOME3'] == cache_dir3a