#
# Copyright 2008-2014 Universidad Complutense de Madrid
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
Recipe requirements
'''

from __future__ import print_function

import inspect


class NoneFind(object):
    pass


def dict_requirement_lookup(source):
    def lookup(req):
        if req.dest in source:
            return source[req.dest]
        else:
            return NoneFind()
    return lookup


class RequirementParser(object):
    '''RecipeRequirement builder.'''
    def __init__(self, recipe, lookup):
        if not inspect.isclass(recipe):
            recipe = recipe.__class__
        self.rClass = recipe.RecipeRequirements
        self.lc = lookup

    def parse(self):
        '''Build the RecipeRequirement object from available metadata.'''
        parameters = {}

        for req in self.rClass.values():
            value = self.lc(req)
            if not isinstance(value, NoneFind):
                parameters[req.dest] = value
        names = self.rClass(**parameters)

        return names

    def print_requirements(self, pad=''):

        for req in self.rClass.values():
            if req.hidden:
                # I Do not want to print it
                continue
            dispname = req.dest

            if req.optional:
                dispname = dispname + '(optional)'

            if req.default is not None:
                dispname = dispname + '=' + str(req.default)

            print("%s%s [%s]" % (pad, dispname, req.description))
