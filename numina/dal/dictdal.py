#
# Copyright 2015-2017 Universidad Complutense de Madrid
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

"""DAL for dictionary-based database of products."""

import logging
import json
from itertools import chain

import numina.store.gtc.load as gtcload
from numina.core import obsres_from_dict
from numina.store import load
from numina.exceptions import NoResultFound
from .absdal import AbsDrpDAL
from .stored import ObservingBlock
from .stored import StoredProduct, StoredParameter
from .diskfiledal import build_product_path
from .utils import tags_are_valid


_logger = logging.getLogger("numina.dal.dictdal")


class BaseDictDAL(AbsDrpDAL):
    def __init__(self, drps, ob_table, prod_table, req_table, extra_data=None):
        super(BaseDictDAL, self).__init__(drps)

        # Check that the structure of the base is correct
        self.ob_table = ob_table
        self.prod_table = prod_table
        self.req_table = req_table
        self.extra_data = extra_data if extra_data else {}

    def search_oblock_from_id(self, obsid):
        try:
            ob = self.ob_table[obsid]
            return ObservingBlock(**ob)
        except KeyError:
            raise NoResultFound("oblock with id %d not found" % obsid)

    def search_prod_obsid(self, ins, obsid, pipeline):
        """Returns the first coincidence..."""
        ins_prod = self.prod_table[ins]

        # search results of these OBs
        for prod in ins_prod:
            if prod['ob'] == obsid:
                # We have found the result, no more checks
                return StoredProduct(**prod)
        else:
            raise NoResultFound('result for ob %i not found' % obsid)

    def search_prod_req_tags(self, req, ins, tags, pipeline):
        if req.dest in self.extra_data:
            val = self.extra_data[req.dest]
            content = load(req.type, val)
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
        for prod in self.prod_table[ins]:
            pk = prod['type'] 
            pt = prod['tags']
            if ((pk == label) or (pk == label_alt)) and tags_are_valid(pt, tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)
                rprod['content'] = load(tipo, prod['content'])
                return StoredProduct(**rprod)
        else:
            msg = 'type %s compatible with tags %r not found' % (tipo, tags)
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
            raise NoResultFound("No parameters for %s mode, pipeline %s", mode, pipeline)

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
                    value = load(req.type, prod['content'])
                    content = StoredParameter(value)
                    return content
            else:
                msg = 'name %s compatible with tags %r not found' % (req.dest, tags)
                raise NoResultFound(msg)

    def obsres_from_oblock_id(self, obsid, configuration=None):
        """"
        Override instrument configuration if configuration is not None
        """
        este = self.ob_table[obsid]
        obsres = obsres_from_dict(este)

        this_drp = self.drps.query_by_name(obsres.instrument)

        for mode in this_drp.modes:
            if mode.key == obsres.mode:
                tagger = mode.tagger
                break
        else:
            raise ValueError('no mode for %s in instrument %s' % (obsres.mode, obsres.instrument))

        if tagger is None:
            master_tags = {}
        else:
            master_tags = tagger(obsres)

        obsres.tags = master_tags

        if configuration:
            # override instrument configuration
            obsres.configuration = self.search_instrument_configuration(obsres.instrument, configuration)
        else:
            # Insert Instrument configuration
            obsres.configuration = this_drp.configuration_selector(obsres)

        return obsres

    def search_result_id(self, child_id, tipo, field):
        pass

    def search_product(self, name, tipo, obsres, options=None):
        # returns StoredProduct
        ins = obsres.instrument
        tags = obsres.tags
        pipeline = obsres.pipeline

        if name in self.extra_data:
            val = self.extra_data[name]
            content = load(tipo, val)
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
                    value = load(tipo, prod['content'])
                    content = StoredParameter(value)
                    return content
            else:
                msg = 'name %s compatible with tags %r not found' % (name, tags)
                raise NoResultFound(msg)

    def search_result_relative(self, name, tipo, obsres, mode, field, node, options=None):
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
    def __init__(self, drps, obtable, base, extra_data=None):

        prod_table = base['products']

        if 'parameters' in base:
            req_table = base['parameters']
        else:
            req_table = base['requirements']

        super(Dict2DAL, self).__init__(drps, obtable, prod_table, req_table, extra_data)


class HybridDAL(Dict2DAL):
    def __init__(self, drps, obtable, base, extra_data=None):
        self.rootdir = base.get("rootdir", "")

        super(HybridDAL, self).__init__(drps, obtable, base, extra_data)

    def search_product(self, name, tipo, obsres, options=None):
        if name in self.extra_data:
            val = self.extra_data[name]
            content = load(tipo, val)
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
        for prod in self.prod_table[instrument]:
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
                rprod['content'] = load(tipo, path)
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

    def search_result_relative(self, name, tipo, obsres, mode, field, node, options=None):
        instrument = obsres.instrument
        if mode is None:
            mode = obsres.mode
        print('search relative', instrument, mode, field, node)

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
            return load(tipo, path)
        else:
            # GTC load
            with open(path) as fd:
                data = json.load(fd)
                inter = gtcload.build_result(data)
                elem = inter['elements']
                return elem[name]

