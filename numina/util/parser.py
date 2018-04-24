#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

#
# Copyright 2017 Universidad Complutense de Madrid
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
        tok = "'{}': {}".format(key, val)
        result.append(tok)
    tokj = ','.join(result)
    result = "{{ {0} }}".format(tokj)
    state = ast.literal_eval(result)
    return state
