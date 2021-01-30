#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import ast


def split_type_name(name):
    fp = name.find('(')
    sp = name.find(')')
    if (fp == -1) or (sp == -1) or (sp < fp):
        # paren not found
        klass = name
        fargs = ""
    else:
        klass = name[:fp]
        fargs = name[fp + 1:sp]
    return klass, fargs


def parse_arg_line(fargs):
    """parse limited form of arguments of function

    in the form a=1, b='c'
    as a dictionary
    """

    # Convert to literal dict
    fargs = fargs.strip()
    if fargs == '':
        return {}

    pairs = [s.strip() for s in fargs.split(',')]
    # find first "="
    result = []
    for p in pairs:
        fe = p.find("=")
        if fe == -1:
            # no equal
            raise ValueError("malformed")
        key = p[:fe]
        val = p[fe + 1:]
        tok = f"'{key}': {val}"
        result.append(tok)
    tokj = ','.join(result)
    result = f"{{ {tokj} }}"
    state = ast.literal_eval(result)
    return state
