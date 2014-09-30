#
# Copyright 2014 Universidad Complutense de Madrid
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

import pickle
import os
import shutil

from numina.user.xdgdirs import xdg_cache_home


class DiskCache(object):
    def __init__(self, cache_dir):
        self._cache = {}
        self.cache_dir = cache_dir
        self.INDEX_MAP = 'index.map'
        self.index_map = os.path.join(self.cache_dir, self.INDEX_MAP)

    def load(self):
        if not os.path.isdir(self.cache_dir):
            # print 'create', self.cache_dir
            os.mkdir(self.cache_dir)

        try:
            print self.index_map
            with open(self.index_map) as pk:
                try:
                    self._cache = pickle.load(pk)
                except EOFError:
                    self._cache = {}
        except IOError:  # File does not exist?
            self._cache = {}

#        print 'loaded cache'

    def update_map(self):
        with open(self.index_map, 'w+') as pk:
            pickle.dump(self._cache, pk)

    def url_is_cached(self, urldigest):
        full = os.path.join(self.cache_dir, urldigest)
        if urldigest in self._cache and os.path.isfile(full):
            return True
        return False

    def cached_filename(self, urldigest):
        return os.path.join(self.cache_dir, urldigest)

    def retrieve(self, urldigest):
        return self._cache[urldigest]

    def update(self, urldigest, filename, etag):
        # Store in cache...
        self._cache[urldigest] = etag
        full = os.path.join(self.cache_dir, urldigest)
        shutil.copyfile(filename, full)


class NuminaDiskCache(DiskCache):
    def __init__(self):
        cache_dir = os.path.join(xdg_cache_home, 'numina')
        super(NuminaDiskCache, self).__init__(cache_dir)
