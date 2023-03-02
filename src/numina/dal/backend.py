#
# Copyright 2015-2021 Universidad Complutense de Madrid
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
import operator

import numina.store
from numina.exceptions import NoResultFound
from numina.util.fqn import fully_qualified_name
from numina.util.context import working_directory

from .dictdal import BaseHybridDAL
from .stored import StoredProduct, StoredResult


_logger = logging.getLogger(__name__)


class BackendTable(object):
    def __init__(self):
        self.table_contents = {}
        self.table_index = []

    def new_id(self):
        if self.table_index:
            newidx = self.table_index[-1] + 1
        else:
            newidx = 1
        self.table_index.append(newidx)
        return newidx

    def insert(self, prodid, prod_reg):
        self.table_contents[prodid] = prod_reg


class Backend(BaseHybridDAL):

    def __init__(self, drps, base, extra_data=None, basedir=None, components=None,
                 filename=None):

        self.filename = filename
        temp_ob_ids = []
        obtable = base.get('oblocks', {})
        if isinstance(obtable, dict):
            for ob in obtable:
                temp_ob_ids.append(ob)
            obdict = obtable
        elif isinstance(obtable, list):
            obdict = {}
            for ob in obtable:
                temp_ob_ids.append(ob['id'])
                obdict[ob['id']] = ob
        else:
            raise TypeError("'oblocks' must be a list or a dictionary")

        self.db_tables = {}

        self.db_tables['tasks'] = base.get('tasks', {})
        self.db_tables['tasks_index'] = base.get('tasks_index', [])

        self.db_tables['results'] = base.get('results', {})
        self.db_tables['results_index'] = base.get('results_index', [])

        self.db_tables['products'] = base.get('products', {})
        self.db_tables['products_index'] = base.get('products_index', [])

        super(Backend, self).__init__(
            drps, obdict, base,
            extra_data=extra_data,
            basedir=basedir,
            components=components
        )
        
        self.ob_ids = temp_ob_ids
        self.db_tables['oblocks'] = self.ob_table
        self.db_tables['requirements'] = self.req_table
        self.db_tables['oblocks_index'] = self.ob_ids
        self.prod_table = self.db_tables['products']

    def dump_data(self):
        state = {}
        state['version'] = 2
        database = {}
        state['database'] = database
        for name, val in self.db_tables.items():
            database[name] = val
        return state

    def dump_to_file(self):
        if self.filename:
            with open(self.filename, 'w') as fp:
                self.dump(fp)

    def new_id(self, table_index):
        if table_index:
            newidx = table_index[-1] + 1
        else:
            newidx = 1
        table_index.append(newidx)
        self.dump_to_file()
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
        self.dump_to_file()
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
        self.dump_to_file()

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
            'uuid': str(task.result.uuid),
            'qc': task.result.qc.name,
            'mode': task.request_runinfo['mode'],
            'instrument': task.request_runinfo['instrument'],
            'pipeline': task.request_runinfo['pipeline'],
            'time_create': task.time_end.strftime('%FT%T'),
            'time_obs': '',
            'recipe_class': task.request_runinfo['recipe_class'],
            'recipe_fqn': task.request_runinfo['recipe_fqn'],
            'oblock_id': task.request_params['oblock_id'],
            'result_dir': res_dir,
            'result_file': filename
        }
        self.db_tables['results'][newix] = result_reg

        for key, prod in result.stored().items():

            DB_PRODUCT_KEYS = [
                'instrument',
                'observation_date',
                'uuid',
                'quality_control'
            ]

            if prod.type.isproduct():
                newprod = self.new_product_id()
                _logger.debug('product_id=%d in backend', newprod)
                internal_value = getattr(result, key)
                ometa = prod.type.extract_db_info(internal_value, DB_PRODUCT_KEYS)
                prod_reg = {
                    'id': newprod,
                    'result_id': newix,
                    'qc': task.result.qc.name,
                    'instrument': task.request_runinfo['instrument'],
                    'time_create': task.time_end.strftime('%FT%T'),
                    'time_obs': ometa['observation_date'].strftime('%FT%T'),
                    'tags': ometa['tags'],
                    'uuid': ometa['uuid'],
                    'oblock_id': task.request_params['oblock_id'],
                    'type': prod.type.name(),
                    'type_fqn': fully_qualified_name(prod.type),
                    'content': os.path.join(res_dir, serialized['values'][key])
                }
                self.db_tables['products'][newprod] = prod_reg
        self.dump_to_file()

    def _search_prod_table(self, name, tipo, obsres):
        """Returns the first coincidence..."""

        instrument = obsres.instrument

        conf = str(obsres.configuration.origin.uuid)

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
        pred_func = tipo.query_expr.fill_placeholders(**obsres.tags)
        for prod in iter_over:
            pi = prod['instrument']
            pk = prod['type']
            pt = prod['tags']

            if pi == instrument and ((pk == label) or (pk == label_alt)):
                pred_val = pred_func.eval(**pt)
                # pred_val must be boolean to check if it is valid
                # if the expression is incomplete it may be non-bool
                if pred_val:
                    # this is a valid product
                    # We have found the result, no more checks
                    # Make a copy
                    rprod = dict(prod)
                    path = prod['content']
                    # Check if path is absolute
                    if not os.path.isabs(path):
                        path = os.path.join(self.basedir, path)
                    rprod['content'] = numina.store.load(tipo, path)
                    return StoredProduct(**rprod)
        else:
            msg = f'type {label_alt} compatible with tags {obsres.tags} not found'
            raise NoResultFound(msg)

    def search_result_id(self, node_id, tipo, field, mode=None):
        cobsres = self.obsres_from_oblock_id(node_id)

        if mode is not None:
            # mode must match
            if cobsres.mode != mode:
                msg = f"requested mode '{mode}' and obsmode '{cobsres.mode}' do not match"
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
                filename = result_reg['result_file']
                # change directory to open result file
                with working_directory(os.path.join(self.basedir, directory)):
                    with open(filename) as fd:
                        data = json.load(fd)
                        stored_result = StoredResult.load_data(data)
                try:
                    content = getattr(stored_result, field)
                except AttributeError:
                    raise NoResultFound(f'no field {field} found in result')

                st = StoredProduct(
                    id=result_reg['id'],
                    content=content,
                    tags={}
                )
                return st
            else:
                msg = f"result of mode '{field}' id={node_id} not found"
                raise NoResultFound(msg)
        except KeyError as err:
            msg = f"field '{field}' not found in result of mode '{cobsres.mode}' id={node_id}"
            raise NoResultFound(msg) from err

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
        result_file = result_reg['result_file']
        result_dir = result_reg.get('result_dir', '')

        with working_directory(os.path.join(self.basedir, result_dir)):
            with open(result_file) as fd:
                import json
                data = json.load(fd)
                return StoredResult.load_data(data)
