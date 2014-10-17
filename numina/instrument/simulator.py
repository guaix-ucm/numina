#
# Copyright 2011-2014 Universidad Complutense de Madrid
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

'''
    Components for instrument simulation
'''

import logging

_logger = logging.getLogger('numina.simulator')


class Grism(object):
    def __init__(self, description, cid=0):
        self.name = description.name


class InstrumentWheel(object):
    def __init__(self, description, cid=0):

        self.cid = cid
        self.fwpos = 0
        self.fwmax = len(description.grisms)

        self.elements = []

        for cid, grism in enumerate(description.grisms):
            el = Grism(grism, cid=cid)
            self.elements.append(el)

    def turn(self, position):
        self.fwpos += (position % self.fwmax)
        return self.fwpos

    def set_position(self, position):
        self.fwpos = (position % self.fwmax)
        return self.fwpos

    def illum(self, ls):
        return ls

    def current(self):
        return self.elements[self.fwpos]

    def current_element(self):
        return self.current()._object_path
