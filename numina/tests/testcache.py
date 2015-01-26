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


from .diskcache import NuminaDiskCache
from .download import download_cache as base_download
from .download import BLOCK


# Initialize global cache
testcache = NuminaDiskCache()
testcache.load()


def download_cache(url, bsize=BLOCK):
    return base_download(url, testcache, bsize=bsize)
