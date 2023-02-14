#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""User command line interface of Numina."""

import warnings

import yaml

import numina
import numina.drps
from numina.user.clishowins import print_no_instrument


def register(subparsers, config):
    parser_show_rec = subparsers.add_parser(
        'show-recipes',
        help='show information of recipes'
        )

    parser_show_rec.set_defaults(command=show_recipes, template=False)

    parser_show_rec.add_argument(
        '-i', '--instrument',
        help='filter recipes by instrument'
        )
    parser_show_rec.add_argument(
        '-m', '--mode',
        help='filter recipes by mode name'
    )
#    parser_show_rec.add_argument('--output', type=argparse.FileType('wb', 0))

    parser_show_rec.add_argument(
        'name', nargs='*', default=None,
        help='filter recipes by name'
        )

    return parser_show_rec


def show_recipes(args, extra_args):

    drpsys = numina.drps.get_system_drps()

    # Query instruments
    if args.instrument:
        name = args.instrument
        try:
            val = drpsys.query_by_name(name)
        except KeyError:
            val = None
        res = [(name, val)]
    else:
        res = drpsys.query_all().items()

    # predicates
    preds = []
    if args.name:
        pred1 = lambda mode_rec: mode_rec[1]['class'] in args.name
        preds.append(pred1)
    if args.mode:
        pred1 = lambda mode_rec: mode_rec[0] == args.mode
        preds.append(pred1)

    for name, theins in res:
        # Per instrument
        if theins:
            for pipe in theins.pipelines.values():
                for mode, recipe_entry in pipe.recipes.items():
                    mod_rec = mode, recipe_entry
                    # Check all predicates
                    for pre in preds:
                        if not pre(mod_rec):
                            break
                    else:
                        recipe_fqn = recipe_entry['class']
                        recipe = pipe.get_recipe_object(mode)
                        print_recipe(
                            recipe.__class__, name=recipe_fqn,
                            insname=theins.name,
                            pipename=pipe.name,
                            modename=mode
                            )
        else:
            print_no_instrument(name)


def print_requirements(recipe, pad=''):

    for req in recipe.requirements().values():
        if req.hidden:
            # I Do not want to print it
            continue
        dispname = req.dest

        if req.optional:
            dispname = dispname + '(optional)'

        if req.default is not None:
            dispname = dispname + '=' + str(req.default)
        typ = req.type.descriptive_name()

        print(f"{pad}{dispname} type={typ!r} [{req.description}]")


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
        warnings.warn(f'problem {error} with recipe {recipe!r}')
