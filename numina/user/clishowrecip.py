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

'''User command line interface of Numina.'''

from __future__ import print_function

import logging

import yaml

from numina import __version__
from numina.core import DataFrameType, DataProductType
from numina.core import import_object


_logger = logging.getLogger("numina")


def show_recipes(args):
    this_recipe_print = print_recipe
    if args.template:
        this_recipe_print = print_recipe_template

    for theins in args.drps.values():
        # Per instrument
        if not args.instrument or (args.instrument == theins.name):
            for pipe in theins.pipelines.values():
                for mode, recipe_fqn in pipe.recipes.items():
                    if not args.name or (recipe_fqn in args.name):
                        Cls = import_object(recipe_fqn)
                        this_recipe_print(
                            Cls, name=recipe_fqn,
                            insname=theins.name,
                            pipename=pipe.name,
                            modename=mode
                            )


def print_recipe_template(recipe, name=None, insname=None,
                          pipename=None, modename=None):

    def print_io(req):
        dispname = req.dest
        if getattr(req, 'default', None) is not None:
            return (dispname, req.default)
        elif isinstance(req.type, DataFrameType):
            return (dispname, dispname + '.fits')
        elif isinstance(req.type, DataProductType):
            return (dispname, getattr(req.type, 'default', None))
        else:
            return (dispname, None)

    # Create a dictionary with templates
    requires = {}
    optional = {}
    for req in recipe.Requirements.values():
        if req.hidden:
            # I Do not want to print it
            continue
        if req.optional:
            out = optional
        else:
            out = requires

        k, v = print_io(req)
        out[k] = v

    final = dict(requirements=requires)

    print('# This is a numina %s template file' % (__version__,))
    print('# for recipe %r' % (name,))
    print('#')
    if optional:
        print('# The following requirements are optional:')
        for kvals in optional.items():
            print('#  %s: %s' % kvals)
        print('# end of optional requirements')
    print(yaml.dump(final), end='')
    print('#products:')
    for prod in recipe.Result.values():
        print('# %s: %s' % print_io(prod))
    print('#logger:')
    print('# logfile: processing.log')
    print('# format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"')
    print('# enabled: true')
    print('---')


def print_requirements(recipe, pad=''):

    for req in recipe.Requirements.values():
        if req.hidden:
            # I Do not want to print it
            continue
        dispname = req.dest

        if req.optional:
            dispname = dispname + '(optional)'

        if req.default is not None:
            dispname = dispname + '=' + str(req.default)
        typ = req.type.python_type.__name__

        print("%s%s type=%r [%s]" % (pad, dispname, typ, req.description))


def print_recipe(recipe, name=None, insname=None,
                 pipename=None, modename=None):
    try:
        if name is None:
            name = recipe.__module__ + '.' + recipe.__name__
        print('Recipe:', name)
        if recipe.__doc__:
            print(' summary:',
                  recipe.__doc__.lstrip().expandtabs().splitlines()[0]
                  )
        if insname:
            print(' instrument:', insname)
        if pipename:
            print('  pipeline:', pipename)
        if modename:
            print('  obs mode:', modename)
        print(' requirements:')
        print_requirements(recipe, pad='  ')
        print()
    except Exception as error:
        _logger.warning('problem %s with recipe %r', error, recipe)

