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

'''
Checking the requirements
'''
from numina.core import DataFrame
from numina.core.oresult import ObservationResult


class KeyChecker(object):
    def __init__(self, only=None, keys=None):
        self.only = [] if only is None else only
        self.keys = [] if keys is None else keys

    def check(self, ri):
        images = []
        keyvales = []
        for key in self.only:
            val = getattr(ri, key)
            if isinstance(val, DataFrame):
                images.append(val)
            elif isinstance(val, ObservationResult):
                images.extend(val.frames)
            else:
                raise TypeError('Dunno what to do')

        for image in images:
            with image.open() as hdul:
                header0 = hdul[0].header
                thisk = []
                for key in self.keys:
                    thisk.append(header0[key])
                keyvales.append(thisk)

        if any(x != keyvales[0] for x in keyvales):
            raise TypeError('Errorororor')
        return True


class keycheck(object):
    '''Decorate a RecipeRequirement to check the observation result.

        ::

        from numina.core.checkers import keyckeck

        @keyckeck(only=['darkframe'], keys=['key1', 'key'])
        class Req(RecipeRequirement):
            obresult = ObservationResultRequirement()
            darkframe = Product(MasterDark)

    '''
    def __init__(self, only=None, keys=None):
        self.checker = KeyChecker(only=only, keys=keys)

    def __call__(self, klass):
        __checkers__ = getattr(klass, '__checkers__', [])
        __checkers__.append(self.checker)
        klass.__checkers__ = __checkers__
        return klass
