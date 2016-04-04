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


class StoredProduct(object):
    def __init__(self, id, content, tags, **kwds):
        self.id = id
        self.content = content
        self.tags = tags


class StoredParameter(object):
    def __init__(self, content):
        self.content = content


# A translation of the entries of oblocks
# Notice that this is different to ObservationResult
# that contains the results of the reductions
class ObservingBlock(object):
    def __init__(self, id, instrument, mode, images, children, parent):
        self.id = id
        self.instrument = instrument
        self.mode = mode
        # only one of files and children can
        # be different from []
        self.images = images
        self.frames = self.images
        self.children = children
        self.parent = parent
