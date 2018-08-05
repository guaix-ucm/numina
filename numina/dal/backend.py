#
# Copyright 2015-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DAL for dictionary-based database of products."""

import os
import logging
import json
from itertools import chain

import operator

import six
import numina.store
import numina.store.gtc.load as gtcload

from numina.exceptions import NoResultFound
from numina.util.fqn import fully_qualified_name
from numina.util.objimport import import_object
from numina.util.context import working_directory
from numina.types.qc import QC

from .dictdal import Dict2DAL
from .stored import StoredProduct
from .diskfiledal import build_product_path
from .utils import tags_are_valid


_logger = logging.getLogger(__name__)


# FIXME: this is already implemented, elsewhere
# There should be one-- and preferably only one --obvious way to do it.

def is_fits(filename, **kwargs):
    return filename.endswith('.fits')


def read_fits(filename):
    import numina.types.dataframe as df
    import astropy.io.fits as fits
    return df.DataFrame(frame=fits.open(filename))


def is_json(filename, **kwargs):
    return filename.endswith('.json')


def read_json(filename):
    import json

    with open(filename) as fd:
        base = json.load(fd)
    return read_structured(base)


def is_yaml(filename, **kwargs):
    return filename.endswith('.yaml')


def read_yaml(filename):
    import yaml

    with open(filename) as fd:
        base = yaml.load(fd)

    return read_structured(base)


def read_structured(data):

    if 'type_fqn' in data:
        type_fqn = data['type_fqn']
        cls = import_object(type_fqn)
        obj = cls.__new__(cls)
        obj.__setstate__(data)
        return obj
    return data


def unserial(value):
    checkers = [(is_fits, read_fits), (is_json, read_json), (is_yaml, read_yaml)]
    if isinstance(value, six.string_types):
        for check_type, conv in checkers:
            if check_type(value):
                return conv(value)
        else:
            return value
    else:
        return value


class StoredResult(object):
    """Recover the RecipeResult values stored in the Backend"""
    def __init__(self):
        self.qc = QC.UNKNOWN

    @classmethod
    def load_data(cls, state):
        obj = cls.__new__(cls)
        obj._from_dict(state)
        return obj
    
    def _from_dict(self, state):
        self.qc = QC[state.get('qc', 'UNKNOWN')]
        values = state.get('values', {})
        if isinstance(values, list):
            values = {o['name']: o['content'] for o in values}
        for key, val in values.items():
            loaded = unserial(val)
            setattr(self, key, loaded)


class Backend(Dict2DAL):

    # FIXME: most code here is duplicated with HybridDal
    def __init__(self, drps, base, extra_data=None, basedir=None):

        self.rootdir = base.get("rootdir", "")
        self.ob_ids = []

        if basedir is None:
            self.basedir = os.getcwd()
        else:
            self.basedir = basedir

        obtable = base.get('oblocks', {})
        if isinstance(obtable, dict):
            for ob in obtable:
                self.ob_ids.append(ob)
            obdict = obtable
        elif isinstance(obtable, list):
            obdict = {}
            for ob in obtable:
                self.ob_ids.append(ob['id'])
                obdict[ob['id']] = ob
        else:
            raise TypeError

        self.db_tables = {}

        self.db_tables['tasks'] = base.get('tasks', {})
        self.db_tables['tasks_index'] = base.get('tasks_index', [])

        self.db_tables['results'] = base.get('results', {})
        self.db_tables['results_index'] = base.get('results_index', [])

        self.db_tables['products'] = base.get('products', {})
        self.db_tables['products_index'] = base.get('products_index', [])

        super(Backend, self).__init__(drps, obdict, base, extra_data)

        self.db_tables['oblocks'] = self.ob_table
        self.db_tables['requirements'] = self.req_table
        # self.db_tables['obsi'] = self.ob_ids

    def dump_data(self):
        state = {}
        state['version'] = 2
        state['rootdir'] = self.rootdir
        database = {}
        state['database'] = database
        for name, val in self.db_tables.items():
            database[name] = val
        return state

    @staticmethod
    def new_id(table_index):
        if table_index:
            newidx = table_index[-1] + 1
        else:
            newidx = 1
        table_index.append(newidx)
        return newidx

    def new_task_id(self, request, request_params):
        newidx = self.new_id(self.db_tables['tasks_index'])
        return newidx

    def new_result_id(self):
        newidx = self.new_id(self.db_tables['results_index'])
        return newidx
    
    def new_product_id(self):
        newidx = self.new_id(self.db_tables['products_index'])
        return newidx

    def new_task(self, request, request_params):

        task = super(Backend, self).new_task(request, request_params)

        task_reg = {
            'id': task.id, 'state': task.state,
            'time_create': task.time_create.strftime('%FT%T'),
            'request': request,
            'request_params': request_params,
            'request_runinfo': task.request_runinfo
        }
        _logger.debug('insert task=%d in backend', task.id)
        self.db_tables['tasks'][task.id] = task_reg

        return task

    def update_task(self, task):
        _logger.debug('update task=%d in backend', task.id)
        task_reg = self.db_tables['tasks'][task.id]
        task_reg['state'] = task.state
        task_reg['time_start'] = task.time_start.strftime('%FT%T')
        task_reg['time_end'] = task.time_end.strftime('%FT%T')
        task_reg['request'] = task.request
        task_reg['request_params'] = task.request_params
        task_reg['request_runinfo'] = task.request_runinfo

    def update_result(self, task, serialized, filename):
        _logger.debug('update result of task=%d in backend', task.id)
        newix = self.new_result_id()
        _logger.debug('result_id=%d in backend', newix)
        result = task.result
        if result is None:
            return

        res_dir = task.request_runinfo['results_dir']
        result_reg = {
            'id': newix,
            'task_id': task.id,
            # 'values': [],
            'qc': task.result.qc.name,
            'mode': task.request_runinfo['mode'],
            'instrument': task.request_runinfo['instrument'],
            'time_create': task.time_end.strftime('%FT%T'),
            'time_obs': '',
            'recipe_class': task.request_runinfo['recipe_class'],
            'recipe_fqn': task.request_runinfo['recipe_fqn'],
            'oblock_id': task.request_params['oblock_id'],
            'result_dir': res_dir,
            'result_file': os.path.join(res_dir, filename)
        }
        self.db_tables['results'][newix] = result_reg


        for key, prod in result.stored().items():
            if prod.dest == 'qc':
                continue

            # This is the same contained in result_file
            # val = {}
            # val['name'] = prod.dest
            # val['type'] = prod.type.name()
            # val['type_fqn'] = fully_qualified_name(prod.type)
            # val['content'] = serialized['values'][key]
            # result_reg['values'].append(val)

            if prod.type.isproduct():
                newprod = self.new_product_id()
                _logger.debug('product_id=%d in backend', newprod)
                internal_value = getattr(result, key)
                origin_metadata = internal_value.meta['origin']
                prod_reg = {
                    'id': newprod,
                    'result_id': newix,
                    'qc': task.result.qc.name,
                    'instrument': task.request_runinfo['instrument'],
                    'time_create': task.time_end.strftime('%FT%T'),
                    'time_obs': origin_metadata['observation_date'].strftime('%FT%T'),
                    'tags': internal_value.meta['tags'],
                    'oblock_id': task.request_params['oblock_id'],
                    'type': prod.type.name(),
                    'type_fqn': fully_qualified_name(prod.type),
                    'content': os.path.join(res_dir, serialized['values'][key])
                }
                self.db_tables['products'][newprod] = prod_reg

    def add_obs(self, obtable):

        # Preprocessing
        obdict = {}
        for ob in obtable:
            obid = ob['id']
            if obid not in self.ob_ids:

                self.ob_ids.append(obid)
                obdict[obid] = ob
            else:
                _logger.warning('oblock_id=%s is already in table', obid)
        # Update parents
        for ob in obdict.values():
            children = ob.get('children', [])
            for ch in children:
                obdict[ch]['parent'] = ob['id']

        self.ob_table.update(obdict)

    def search_product(self, name, tipo, obsres, options=None):
        if name in self.extra_data:
            val = self.extra_data[name]
            content = numina.store.load(tipo, val)
            return StoredProduct(id=0, tags={}, content=content)
        else:
            return self._search_prod_table(name, tipo, obsres)

    def _search_prod_table(self, name, tipo, obsres):
        """Returns the first coincidence..."""

        instrument = obsres.instrument

        conf = obsres.configuration.uuid

        drp = self.drps.query_by_name(instrument)
        label = drp.product_label(tipo)
        # Strip () is present
        if label.endswith("()"):
            label_alt = label[:-2]
        else:
            label_alt = label

        # search results of these OBs
        if isinstance(self.prod_table, dict):
            iter_over = self.prod_table.values()
        else:
            iter_over = self.prod_table

        for prod in iter_over:
            pi = prod['instrument']
            pk = prod['type']
            pt = prod['tags']
            if pi == instrument and ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, obsres.tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)

                if 'content' in prod:
                    path = prod['content']
                else:
                    # Build path
                    path = build_product_path(drp, self.rootdir, conf, name, tipo, obsres)
                _logger.debug("path is %s", path)
                # Check if path is absolute
                if not os.path.isabs(path):
                    path = os.path.join(self.basedir, path)
                rprod['content'] = numina.store.load(tipo, path)
                return StoredProduct(**rprod)
        else:
            # Not in table, try file directly
            _logger.debug("%s not in table, try file directly", tipo)
            path = self.build_product_path(drp, conf, name, tipo, obsres)
            _logger.debug("path is %s", path)
            content = self.product_loader(tipo, name, path)
            return StoredProduct(id=0, content=content, tags=obsres.tags)

    def search_result(self, name, tipo, obsres, resultid=None):

        if resultid is None:
            for g in chain([tipo.name()], tipo.generators()):
                if g in obsres.results:
                    resultid = obsres.results[g]
                    break
            else:
                raise NoResultFound("resultid not found")
        prod = self._search_result(name, tipo, obsres, resultid)
        return prod

    def search_previous_obsres(self, obsres, node=None):

        if node is None:
            node = 'prev'

        if node == 'prev-rel':
            # Compute nodes relative to parent
            # unless parent is None, then is equal to prev
            parent_id = obsres.parent
            if parent_id is not None:
                cobsres = self.obsres_from_oblock_id(parent_id)
                subset_ids = cobsres.children
                idx = subset_ids.index(obsres.id)
                return reversed(subset_ids[:idx])
            else:
                return self.search_previous_obsid_all(obsres.id)
        else:
            return self.search_previous_obsid_all(obsres.id)

    def search_previous_obsid_all(self, obsid):
        idx = self.ob_ids.index(obsid)
        return reversed(self.ob_ids[:idx])

    def search_result_id(self, node_id, tipo, field, mode=None):
        cobsres = self.obsres_from_oblock_id(node_id)

        if mode is not None:
            # mode must match
            if cobsres.mode != mode:
                msg = "requested mode '{}' and obsmode '{}' do not match".format(mode, cobsres.mode)
                raise NoResultFound(msg)

        try:
            candidates = []
            for idx, val in self.db_tables['results'].items():
                if node_id == val['oblock_id']:
                    candidates.append(val)

            s_can = sorted(candidates, key=operator.itemgetter('time_create'), reverse=True)
            if s_can:
                result_reg = s_can[0]
                directory = result_reg.get('result_dir', '')
                field_files = result_reg['values']
                for field_entry in field_files:
                    if field_entry['name'] == field:
                        break
                else:
                    raise NoResultFound('no field {} found'.format(field))

                type_fqn = field_entry['type_fqn']

                type_class = import_object(type_fqn)
                type_obj = type_class()

                content = None
                if field_entry:
                    if field_entry['content']:
                        filename = os.path.join(directory, field_entry['content'])
                        content = numina.store.load(type_obj, filename)

                st = StoredProduct(
                    id=result_reg['id'],
                    content=content,
                    tags={}
                )
                return st
            else:
                msg = "result of mode '{}' id={} not found".format(field, node_id)
                raise NoResultFound(msg)
        except KeyError as err:
            msg = "field '{}' not found in result of mode '{}' id={}".format(field, cobsres.mode, node_id)
            # Python 2.7 compatibility
            six.raise_from(NoResultFound(msg), err)
            # raise NoResultFound(msg) from err

    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):

        _logger.debug('search relative result for %s', name)

        # result_type = DataFrameType()
        result_mode = result_desc.mode
        result_field = result_desc.attr
        result_node = result_desc.node

        ignore_fail = result_desc.ignore_fail

        if result_node == 'children':
            # Results are multiple
            # one per children
            _logger.debug('search children nodes of %s', obsres.id)
            results = []
            for c in obsres.children:
                try:
                    st = self.search_result_id(c, tipo, result_field)
                    results.append(st)
                except NoResultFound:
                    if not ignore_fail:
                        raise

            return results
        elif result_node == 'prev' or result_node == 'prev-rel':
            _logger.debug('search previous nodes of %s', obsres.id)

            # obtain previous nodes
            for previd in self.search_previous_obsres(obsres, node=result_node):
                # print('searching in node', previd)
                try:
                    st = self.search_result_id(previd, tipo, result_field, mode=result_mode)
                    return st
                except NoResultFound:
                    pass

            else:
                raise NoResultFound('value not found in any node')
        elif result_node == 'last':
            _logger.debug('search last node of %s', result_mode)

            try:
                st = self.search_result_last(name, tipo, result_desc)
                return st
            except NoResultFound:
                pass
        else:
            msg = 'unknown node type {}'.format(result_node)
            raise TypeError(msg)

    def search_result_last(self, name, tipo, result_desc):
        # FIXME: Implement
        raise NoResultFound

    def _search_result(self, name, tipo, obsres, resultid):
        """Returns the first coincidence..."""

        instrument = obsres.instrument

        conf = obsres.configuration.uuid

        drp = self.drps.query_by_name(instrument)
        label = drp.product_label(tipo)

        # search results of these OBs
        for prod in self.prod_table[instrument]:
            pid = prod['id']
            if pid == resultid:
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)

                if 'content' in prod:
                    path = prod['content']
                else:
                    # Build path
                    path = build_product_path(drp, self.rootdir, conf, name, tipo, obsres)
                _logger.debug("path is %s", path)
                rprod['content'] = self.product_loader(tipo, name, path)
                return StoredProduct(**rprod)
        else:
            msg = 'result with id %s not found' % (resultid, )
            raise NoResultFound(msg)

    def build_product_path(self, drp, conf, name, tipo, obsres):
        path = build_product_path(drp, self.rootdir, conf, name, tipo, obsres)
        return path

    def product_loader(self, tipo, name, path):
        path, kind = path
        if kind == 0:
            return numina.store.load(tipo, path)
        else:
            # GTC load
            with open(path) as fd:
                data = json.load(fd)
                inter = gtcload.build_result(data)
                elem = inter['elements']
                return elem[name]

    def search_session_ids(self):
        for obs_id in self.ob_ids:
            obdict = self.ob_table[obs_id]
            enabled = obdict.get('enabled', True)
            if ((not enabled) or
                    obdict['mode'] in self._RESERVED_MODE_NAMES
            ):
                # ignore these OBs
                continue


            yield obs_id

    def build_recipe_result(self, result_id):
        result_reg = self.db_tables['results'][result_id]
        result_dir = result_reg.get('result_dir', '')
        with working_directory(result_dir):
            return StoredResult.load_data(result_reg)

    def build_recipe_result2(self, result_id):
        result_reg = self.db_tables['results'][result_id]
        result_file = result_reg['result_file']
        result_dir = result_reg.get('result_dir', '')
        with open(result_file) as fd:
            import json
            data = json.load(fd)
            with working_directory(result_dir):
                return StoredResult.load_data(data)