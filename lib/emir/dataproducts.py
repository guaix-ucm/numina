#
# Copyright 2008-2010 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

'''Data products produced by the EMIR pipeline.'''

from emir.instrument.headers import EmirImageCreator

class EmirImage(object):
    def __init__(self, data, variance=None, numbers=None):
        pass
        
class EmirSpectrum(object):
    def __init__(self):
        pass


def create_result(data, variance, exmap):
    creator = EmirImageCreator()
    hdulist = creator.create(data, extensions=
                             [('VARIANCE', variance, None), 
                              ('MAP', exmap, None)])
    return hdulist


def create_raw(data, headers):
    creator = EmirImageCreator()
    hdulist = creator.create(data, headers)
    return hdulist