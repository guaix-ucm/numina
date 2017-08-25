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

"""Load RecipeResults from GTC's result.json files"""

from __future__ import print_function


from numina.core import import_object
from numina.core import DataFrame


def process_node(node):
    """Process a node in result.json structure"""
    value = node['value']
    mname = node['name']
    typeid = node['typeid']
    if typeid == 52: # StructDataValue
        obj = {}
        for el in value['elements']:
            key, val = process_node(el)
            obj[key] = val

        if value['struct_type'] != 'dict':
            # Value is not a dict
            klass = import_object(value['struct_type'])
            newobj = klass.__new__(klass)
            if hasattr(newobj, '__setstate__'):
                newobj.__setstate__(obj)
            else:
                newobj.__dict__ = obj
            obj = newobj

    elif typeid == 90: # StructDataValueList
        obj = []
        for el in value:
            sobj = {}
            for sel in el['elements']:
                key, val = process_node(sel)
                sobj[key] = val
            obj.append(sobj)

    elif typeid == 45: # Frame
        obj = DataFrame(frame=value['path'])
    else:
        obj = value

    return mname, obj


def build_result(data):
    """Create a dictionary with the contents of result.json"""
    more = {}
    for key, value  in data.items():
        if key != 'elements':
            newnode = value
        else:
            newnode = {}
            for el in value:
                nkey, nvalue = process_node(el)
                newnode[nkey] = nvalue

        more[key] = newnode

    return more

if __name__ == '__main__':

    import json

    filename = "result.json"

    with open(filename) as fd:
        data = json.load(fd)

    res = build_result(data)

    print(res)
