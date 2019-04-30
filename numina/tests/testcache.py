#
# Copyright 2015-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Global cache for testing."""

import os
import sys
import tempfile

import numina.util.context as cntx

from astropy.utils import data


def user_cache_dir(appname=None):
    if sys.platform.startswith('java'):
        import platform
        os_name = platform.java_ver()[3][0]
        if os_name.startswith('Windows'): # "Windows XP", "Windows 7", etc.
            system = 'win32'
        elif os_name.startswith('Mac'):
            system = 'darwin'
        else:
            system = 'linux2'
    else:
        system = sys.platform

    if system == 'darwin':
        path = os.path.expanduser('~/Library/Caches')
        if appname:
            path = os.path.join(path, appname)
    else:
        if 'XDG_CACHE_HOME' not in os.environ.keys():
            path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        else:
            if appname in os.environ['XDG_CACHE_HOME']:
                path = os.environ['XDG_CACHE_HOME'].split(appname)[0]
            else:
                path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        if appname:
            path = os.path.join(path, appname)
    if not os.path.exists(os.path.join(path, 'astropy')):
        os.makedirs(os.path.join(path, 'astropy'))
    return path


def download_cache(url, cache=True):
    """Get a tempfile from an URL"""
    cache_dir = user_cache_dir('numina')

    with cntx.environ(XDG_CACHE_HOME=cache_dir):
        fs = open(data.download_file(url, cache=cache), 'rb')
        with tempfile.NamedTemporaryFile(delete=False) as fd:
            block = fs.read()
            while block:
                fd.write(block)
                block = fs.read()

        return fd
