#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

'''Global cache for testing.'''

from tempfile import NamedTemporaryFile
from astropy.utils import data
import os
import sys


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


def download_cache(url):

    os.environ['XDG_CACHE_HOME'] = user_cache_dir('numina')

    fs = open(data.download_file(url, True))
    with NamedTemporaryFile(delete=False) as fd:
        block = fs.read()
        while block:
            fd.write(block)
            block = fs.read()

    return fd

