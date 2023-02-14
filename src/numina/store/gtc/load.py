#
# Copyright 2017-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Load RecipeResults from GTC's result.json files"""

import os.path

import numina.util.objimport as objimp
import numina.types.dataframe as dataframe


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
            klass = objimp.import_object(value['struct_type'])
            newobj = klass.__new__(klass)
            if hasattr(newobj, '__setstate__'):
                newobj.__setstate__(obj)
            else:
                newobj.__dict__ = obj
            obj = newobj
    elif typeid == 9:
        data = value['data']
        dim = value['dimension']
        shape = dim['height'], dim['width']
        obj = data
    elif typeid == 90: # StructDataValueList
        obj = []
        for el in value:
            sobj = {}
            for sel in el['elements']:
                key, val = process_node(sel)
                sobj[key] = val
            obj.append(sobj)

    elif typeid == 45: # Frame
        obj = dataframe.DataFrame(frame=os.path.abspath(value['path']))
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
