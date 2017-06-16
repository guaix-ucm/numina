#
# Copyright 2016-2017 Universidad Complutense de Madrid
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

"""DAL for file-based database of products."""


from __future__ import print_function


import os
import itertools
import logging

from numina.core import import_object
from numina.store import load
from numina.exceptions import NoResultFound
from numina.dal import StoredProduct

_logger = logging.getLogger(__name__)


def _combinations(seq):
    maxl = len(seq)
    for i in range(maxl + 1, -1, -1):
        for j in itertools.combinations(seq, i):
            yield j


def build_product_path(drp, rootdir, conf, name, tipo, ob):
    klass = tipo.__class__
    _logger.info('search %s of type %s', name, tipo)

    try:
        res = drp.query_provides(tipo.__class__)
        label = res.alias
    except ValueError:
        label = tipo.__class__.__name__

    # search results of these OBs
    # build path based in combinations of tags
    for com in _combinations(ob.tags.values()):
        directory = os.path.join(rootdir, ob.instrument, conf, label, *com)
        _logger.debug('try directory %s', directory)
        try:
            files_s = [filename for filename in sorted(os.listdir(directory))]
            _logger.debug("files in directory: %s", files_s)
            for fname in files_s:
                loadpath = os.path.join(directory, fname)
                _logger.debug("check %s", loadpath)
                if os.path.isfile(loadpath):
                    _logger.debug("is regular file %s", loadpath)
                    _logger.info("found %s", loadpath)
                    return loadpath
                else:
                    _logger.debug("is not regular file %s", loadpath)
        except OSError as msg:
            _logger.debug("%s", msg)
    else:
        msg = 'type %s compatible with tags %r not found' % (
            klass, ob.tags)
        _logger.info("%s", msg)
        raise NoResultFound(msg)


DAL_USE_OFFLINE_CALIBS = True


class DiskFileDAL(object):

    def __init__(self, drp, rootdir, *args, **kwargs):
        super(DiskFileDAL, self).__init__()
        self.drp = drp
        self.rootdir = rootdir
        self.conf = 'default'

    def search_product(self, name, tipo, ob):
        klass = tipo.__class__
        print ('Init search ', name, tipo, tipo.__class__)

        try:
            res = self.drp.query_provides(tipo.__class__)
            label = res.alias
        except ValueError:
            label = tipo.__class__.__name__

        # search results of these OBs
        # build path based in combinations of tags
        for com in _combinations(ob.tags.values()):
            directory = os.path.join(self.rootdir, ob.instrument, self.conf, label, *com)
            print('search in', directory)
            try:
                files_s = [filename for filename in sorted(os.listdir(directory))]
                print("files_s", files_s)
                for fname in files_s:
                    loadpath = os.path.join(directory, fname)
                    print("check ", loadpath)
                    if os.path.isfile(loadpath):
                        print("is regular file ", loadpath)
                        if DAL_USE_OFFLINE_CALIBS:
                            content = load(tipo, loadpath)
                        #else:
                            #data = json.load(open(loadpath))
                            #result = process_result(data)
                            #key = self._field_to_extract[klass]
                            #content = result[key]
                            return StoredProduct(id=files_s[-1], content=content, tags=ob.tags)
                        else:
                            print("not ready")
                    else:
                        print("is not regular file ", loadpath)
            except OSError as msg:
                print(msg)
                #msg = 'type %s compatible with tags %r not found' % (klass, ob.tags)
                #raise NoResultFound(msg)
        else:
            msg = 'type %s compatible with tags %r not found' % (
                    klass, ob.tags)
            print(msg)
            raise NoResultFound(msg)


def process_result(data):
    node = dict()
    node["name"] = "reqs",
    node["typeid"] = 52,
    node["typename"] = "StructDataValue"
    node["value"] = {}
    node["value"]['elements'] = data['elements']
    node["value"]['struct_type'] = 'dict'
    obj = process_node(node)
    return obj


def process_node(node):
    if node['typename'] == "StructDataValue":
        obj = {}
        value = node['value']
#        print value['struct_type']
        for el in value['elements']:
            name = el['name']
            obj[name] = process_node(el)
        if value['struct_type'] != 'dict':
            klass = import_object(value['struct_type'])
            ob = klass.__new__(klass)
            if hasattr(ob, '__setstate__'):
                ob.__setstate__(obj)
            else:
                ob.__dict__ = obj
            return ob
        else:
            return obj
    elif node['typename'] == "StructDataValueList":
        obj = []
        value = node['value']
        for el in value:
            sobj = dict()
            for sel in el['elements']:
                name = sel['name']
                sobj[name] = process_node(sel)
            obj.append(sobj)
    else:
        return node['value']

    return obj
