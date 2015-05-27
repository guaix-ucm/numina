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

'''Store backends loader.'''

import logging

import pkg_resources


_logger = logging.getLogger('numina')


def init_store_backends(backend='default'):
    '''Load storage backends.'''

    for entry in pkg_resources.iter_entry_points(group='numina.storage.1'):
        store_loader = entry.load()
        store_loader()


init_dump_backends = init_store_backends

