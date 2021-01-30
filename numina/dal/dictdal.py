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
from itertools import chain

import yaml

import numina.store
import numina.store.gtc.load as gtcload
from numina.core.oresult import obsres_from_dict, oblock_from_dict
from numina.exceptions import NoResultFound
from numina.instrument.assembly import assembly_instrument
from numina.util.context import working_directory

from .absdal import AbsDrpDAL
from .stored import ObservingBlock
from .stored import StoredProduct, StoredParameter, StoredResult
from .diskfiledal import build_product_path
from .utils import tags_are_valid


_logger = logging.getLogger(__name__)


class BaseDictDAL(AbsDrpDAL):
    """A dictionary based DAL"""

    _RESERVED_MODE_NAMES = ['nulo', 'container', 'root', 'raiz']

    def __init__(self, drps, ob_table, prod_table, req_table, extra_data=None, components=None):
        super(BaseDictDAL, self).__init__(drps)
        # Check that the structure of the base is correct
        self.ob_table = ob_table
        self.prod_table = prod_table
        self.req_table = req_table
        self.extra_data = extra_data if extra_data else {}
        self.components = components if components else {}

    def search_oblock_from_id(self, obsid):
        try:
            ob = self.ob_table[obsid]
            return ObservingBlock(**ob)
        except KeyError:
            raise NoResultFound(f"oblock with id {obsid} not found")

    def search_prod_obsid(self, ins, obsid, pipeline):
        """Returns the first coincidence..."""
        ins_prod = self.prod_table.get(ins, [])

        # search results of these OBs
        for prod in ins_prod:
            if prod['ob'] == obsid:
                # We have found the result, no more checks
                return StoredProduct(**prod)
        else:
            raise NoResultFound(f'result for ob {obsid} not found')

    def search_prod_req_tags(self, req, ins, tags, pipeline):
        if req.dest in self.extra_data:
            val = self.extra_data[req.dest]
            content = numina.store.load(req.type, val)
            return StoredProduct(id=0, tags={}, content=content)
        else:
            return self.search_prod_type_tags(req.type, ins, tags, pipeline)

    def search_prod_type_tags(self, tipo, ins, tags, pipeline):
        """Returns the first coincidence..."""

        drp = self.drps.query_by_name(ins)
        label = drp.product_label(tipo)

        # Strip () is present
        if label.endswith("()"):
            label_alt = label[:-2]
        else:
            label_alt = label

        # search results of these OBs
        ptable = self.prod_table.get(ins, [])
        for prod in ptable:
            pk = prod['type'] 
            pt = prod['tags']
            if ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)
                rprod['content'] = numina.store.load(tipo, prod['content'])
                return StoredProduct(**rprod)
        else:
            msg = f'type {tipo} compatible with tags {tags!r} not found'
            raise NoResultFound(msg)

    def search_param_req(self, req, instrument, mode, pipeline):
        req_table_ins = self.req_table.get(instrument, {})
        req_table_insi_pipe = req_table_ins.get(pipeline, {})
        mode_keys = req_table_insi_pipe.get(mode, {})
        if req.dest in self.extra_data:
            value = self.extra_data[req.dest]
            content = StoredParameter(value)
            return content
        elif req.dest in mode_keys:
            value = mode_keys[req.dest]
            content = StoredParameter(value)
            return content
        else:
            raise NoResultFound(f"No parameters for {mode} mode, pipeline {pipeline}")

    def search_param_req_tags(self, req, instrument, mode, tags, pipeline):
        req_table_ins = self.req_table.get(instrument, {})
        req_table_insi_pipe = req_table_ins.get(pipeline, {})
        mode_list = req_table_insi_pipe.get(mode, [])
        if req.dest in self.extra_data:
            value = self.extra_data[req.dest]
            content = StoredParameter(value)
            return content
        else:
            for prod in mode_list:
                pn = prod['name']
                pt = prod['tags']
                if pn == req.dest and tags_are_valid(pt, tags):
                    # We have found the result, no more checks
                    value = numina.store.load(req.type, prod['content'])
                    content = StoredParameter(value)
                    return content
            else:
                msg = f'name {req.dest} compatible with tags {tags!r} not found'
                raise NoResultFound(msg)

    def oblock_from_id(self, obsid):

        este = self.ob_table[obsid]
        oblock = oblock_from_dict(este)

        return oblock

    def obsres_from_oblock_id(self, obsid, as_mode=None, configuration=None):
        """"
        Override instrument configuration if configuration is not None
        """

        oblock = self.oblock_from_id(obsid)

        return self.obsres_from_oblock(oblock, as_mode)

    def obsres_from_oblock(self, oblock, as_mode=None):

        from numina.core.oresult import ObservationResult

        # Internal copy
        obsres = ObservationResult.__new__(ObservationResult)
        obsres.__dict__ = oblock.__dict__

        obsres.mode = as_mode or obsres.mode
        _logger.debug("obsres_from_oblock_2 id='%s', mode='%s' START", obsres.id, obsres.mode)

        try:
            this_drp = self.drps.query_by_name(obsres.instrument)
        except KeyError:
            raise ValueError(f'no DRP for instrument {obsres.instrument}')

        # Reserved names
        if obsres.mode in self._RESERVED_MODE_NAMES:
            selected_mode = None  # null mode
        else:
            selected_mode = this_drp.modes[obsres.mode]

        if selected_mode:
            # This is used if we pass a option here or
            # if mode.build_ob_options is defined in the mode class
            # seems useless
            obsres = selected_mode.build_ob(obsres, self)
            # Not needed, all the information is obtained from
            # the requirements
            obsres = selected_mode.tag_ob(obsres)

        _logger.debug('assembly instrument model')
        key, date_obs, keyname = this_drp.select_profile(obsres)
        obsres.configuration = self.assembly_instrument(key, date_obs, keyname)
        obsres.profile = obsres.configuration

        auto_configure = True
        sample_frame = obsres.get_sample_frame()
        if auto_configure and sample_frame is not None:
            _logger.debug('configuring instrument model with image')
            obsres.profile.configure_with_image(sample_frame.open())
        else:
            _logger.debug('no configuring instrument model')
        return obsres

    def assembly_instrument(self, keyval, date, by_key='name'):
        return assembly_instrument(self.components, keyval, date, by_key=by_key)

    def search_result_id(self, node_id, tipo, field):
        cobsres = self.obsres_from_oblock_id(node_id)

        rdir = resultsdir_default(self.basedir, node_id)
        # FIXME: hardcoded
        taskfile = os.path.join(rdir, 'task.yaml')
        resfile = os.path.join(rdir, 'result.yaml')
        result_contents = yaml.load(open(resfile))
        task_contents = yaml.load(open(taskfile))

        try:
            field_file = result_contents[field]
            st = StoredProduct(
                id=node_id,
                content=numina.store.load(tipo, os.path.join(rdir, field_file)),
                tags={}
            )
            return st
        except KeyError as err:
            msg = f"field '{field}' not found in result of mode '{cobsres.mode}' id={node_id}"
            raise NoResultFound(msg) from err

    def search_product(self, name, tipo, obsres, options=None):
        # returns StoredProduct
        ins = obsres.instrument
        tags = obsres.tags
        pipeline = obsres.pipeline

        if name in self.extra_data:
            val = self.extra_data[name]
            content = numina.store.load(tipo, val)
            return StoredProduct(id=0, tags={}, content=content)
        else:
            return self.search_prod_type_tags(tipo, ins, tags, pipeline)

    def search_parameter(self, name, tipo, obsres, options=None):
        # returns StoredProduct
        instrument = obsres.instrument
        mode = obsres.mode
        tags = obsres.tags
        pipeline = obsres.pipeline

        req_table_ins = self.req_table.get(instrument, {})
        req_table_insi_pipe = req_table_ins.get(pipeline, {})
        mode_list = req_table_insi_pipe.get(mode, [])
        if name in self.extra_data:
            value = self.extra_data[name]
            content = StoredParameter(value)
            return content
        else:
            for prod in mode_list:
                pn = prod['name']
                pt = prod['tags']
                if pn == name and tags_are_valid(pt, tags):
                    # We have found the result, no more checks
                    value = numina.store.load(tipo, prod['content'])
                    content = StoredParameter(value)
                    return content
            else:
                msg = f'name {name} compatible with tags {tags!r} not found'
                raise NoResultFound(msg)

    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):
        # mode field node could go together...
        return []


class DictDAL(BaseDictDAL):
    def __init__(self, drps, base):

        # Check that the structure of 'base' is correct
        super(DictDAL, self).__init__(
            drps,
            base['oblocks'],
            base['products'],
            base['parameters']
        )


class Dict2DAL(BaseDictDAL):
    def __init__(self, drps, obtable, base, extra_data=None, components=None):

        prod_table = base.get('products', {})

        if 'parameters' in base:
            req_table = base['parameters']
        else:
            req_table = base.get('requirements', {})

        super(Dict2DAL, self).__init__(
            drps, obtable, prod_table, req_table,
            extra_data, components=components
        )

    def new_task_id(self, request, request_params):
        if request == 'reduce':
            return request_params.get('oblock_id', 1)
        return 1

    def new_task(self, request, request_params):

        from numina import __version__
        from numina.user.helpers import ProcessingTask

        newidx = self.new_task_id(request, request_params)
        _logger.debug('create task=%s', newidx)
        task = ProcessingTask()
        task.id = newidx
        task.request = request
        task.request_params = request_params
        task.request_runinfo['runner'] = 'numina'
        task.request_runinfo['runner_version'] = __version__
        return task

    def update_task(self, task):
        pass

    def update_result(self, task, serialized, filename):
        pass

    def dump(self, fp):
        state = self.dump_data()
        yaml.dump(state, fp)
        # yaml.dump(state, fp, default_flow_style=False)

        with open('control_dump.json', 'w') as fp:
            json.dump(state, fp, indent=2)

    def dump_data(self):
        state = {}
        state['version'] = 1
        state['products'] = self.prod_table
        state['requirements'] = self.req_table
        state['oblocks'] = self.ob_table
        return state


def workdir_default(basedir, obsid):
    workdir = os.path.join(basedir, f'obsid{obsid}_work')
    workdir = os.path.abspath(workdir)
    return workdir


def resultsdir_default(basedir, obsid):
    resultsdir = os.path.join(basedir, f'obsid{obsid}_results')
    resultsdir = os.path.abspath(resultsdir)
    return resultsdir


class BaseHybridDAL(Dict2DAL):

    def __init__(self, drps, obtable, base, extra_data=None, basedir=None, components=None):

        self.rootdir = base.get("rootdir", "")
        self.ob_ids = []

        if basedir is None:
            self.basedir = os.getcwd()
        else:
            self.basedir = basedir

        super(BaseHybridDAL, self).__init__(
            drps, obtable, base, extra_data=extra_data, components=components
        )

    def add_obs(self, obtable):
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
        raise NotImplementedError

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

    def _search_result(self, name, tipo, obsres, resultid):
        """Returns the first coincidence..."""

        instrument = obsres.instrument

        conf = str(obsres.configuration.origin.uuid)

        drp = self.drps.query_by_name(instrument)
        label = drp.product_label(tipo)

        # search results of these OBs
        ptable = self.prod_table.get(instrument, [])
        for prod in ptable:
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
            msg = f'result with id {resultid} not found'
            raise NoResultFound(msg)

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
        raise NotImplementedError

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
            msg = f'unknown node type {result_node}'
            raise TypeError(msg)

    def search_result_last(self, name, tipo, result_desc):
        # FIXME: Implement
        raise NoResultFound

    def build_product_path(self, drp, conf, name, tipo, obsres):
        path = build_product_path(drp, self.rootdir, conf, name, tipo, obsres)
        return path

    def search_session_ids(self):
        for obs_id in self.ob_ids:
            obdict = self.ob_table[obs_id]
            enabled = obdict.get('enabled', True)
            if (not enabled) or obdict['mode'] in self._RESERVED_MODE_NAMES:
                # ignore these OBs
                continue

            yield obs_id

    def dump_data(self):
        state = super(BaseHybridDAL, self).dump_data()
        state['rootdir'] = self.rootdir
        state['ob_ids'] = self.ob_ids
        return state


class HybridDAL(BaseHybridDAL):
    """A DAL that can read files from directory structure"""

    def __init__(self, drps, obtable, base, extra_data=None, components=None, basedir=None):

        temp_ob_ids = []
        # Preprocessing
        obdict = {}
        for ob in obtable:
            obid = ob['id']
            temp_ob_ids.append(obid)
            obdict[obid] = ob

        # Update parents
        for ob in obdict.values():
            children = ob.get('children', [])
            for ch in children:
                obdict[ch]['parent'] = ob['id']

        super(HybridDAL, self).__init__(
            drps, obdict, base,
            extra_data=extra_data,
            basedir=basedir,
            components=components
        )
        # This field does not exist until super is called
        self.ob_ids = temp_ob_ids

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
        ptable = self.prod_table.get(instrument, [])
        for prod in ptable:
            pk = prod['type']
            pt = prod['tags']
            if ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, obsres.tags):
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
                rprod['content'] = numina.store.load(tipo, path)
                return StoredProduct(**rprod)
        else:
            # Not in table, try file directly
            _logger.debug("%s not in table, try file directly", tipo)
            path = self.build_product_path(drp, conf, name, tipo, obsres)
            _logger.debug("path is %s", path)
            content = self.product_loader(tipo, name, path)
            return StoredProduct(id=0, content=content, tags=obsres.tags)

    def search_result_id(self, node_id, tipo, field, mode=None):
        cobsres = self.obsres_from_oblock_id(node_id)

        if mode is not None:
            # mode must match
            if cobsres.mode != mode:
                msg = f"requested mode '{mode}' and obsmode '{cobsres.mode}' do not match"
                raise NoResultFound(msg)

        try:
            directory = resultsdir_default(self.basedir, node_id)

            # change directory to open result file
            with working_directory(os.path.join(self.basedir, directory)):

                # Try to open both
                filename_yaml = 'result.yaml'
                filename_json = 'result.json'
                if os.path.exists(filename_yaml):
                    with open(filename_yaml) as fd:
                        result_data = yaml.safe_load(fd)
                elif os.path.exists(filename_json):
                    with open(filename_json) as fd:
                        result_data = json.load(fd)
                else:
                    raise ValueError(f'result.yaml or result.json not found in {directory}')

                stored_result = StoredResult.load_data(result_data)

                try:
                    content = getattr(stored_result, field)
                except AttributeError:
                    raise NoResultFound(f'no field {field} found in result')

                st = StoredProduct(
                    id=node_id,
                    content=content,
                    tags={}
                )
                return st
        except KeyError as err:
            msg = f"field '{field}' not found in result of mode '{cobsres.mode}' id={node_id}"
            raise NoResultFound(msg) from err
