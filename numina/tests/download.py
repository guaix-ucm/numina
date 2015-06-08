#
# Copyright 2014-2015 Universidad Complutense de Madrid
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

import hashlib
from tempfile import NamedTemporaryFile

import six
from six.moves import urllib_request
from six.moves import urllib_error


BLOCK = 2048


def download(url, bsize=BLOCK):
    req = urllib_request.Request(url)
    source = urllib_request.urlopen(req)
    #
    assert False
    with NamedTemporaryFile(delete=False) as fd:
        block = source.read(bsize)
        while block:
            fd.write(block)
            block = source.read(bsize)

    return fd


def download_cache(url, cache, bsize=BLOCK):
    hh = hashlib.md5()
    hh.update(six.b(url))
    urldigest = hh.hexdigest()
    update_cache = False
    # print('urldigest', urldigest)
    # print(cache._cache)
    if cache.url_is_cached(urldigest):
        # Retrieve from cache
        etag = cache.retrieve(urldigest)
        # print('is in cache, etag is', etag)
        req = urllib_request.Request(url)
        req.add_header('If-None-Match', etag)
    else:
        # print('resource not in cache')
        req = urllib_request.Request(url)
    try:
        source = urllib_request.urlopen(req)
        update_cache = True
        etag = source.headers['etag']
    except urllib_error.HTTPError as err:
        if err.code == 304:
            update_cache = False
            source = open(cache.cached_filename(urldigest), mode='rb')
        else:
            raise

    #
    with NamedTemporaryFile(delete=False) as fd:
        block = source.read(bsize)
        while block:
            fd.write(block)
            block = source.read(bsize)

    if update_cache:
        # print 'updating cache'
        cache.update(urldigest, fd.name, etag)
        cache.update_map()

    return fd
