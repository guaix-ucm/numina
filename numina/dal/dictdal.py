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

from numina.core import import_object
from numina.core import fully_qualified_name
from numina.core import obsres_from_dict
from numina.store import load
from numina.exceptions import NoResultFound
from .absdal import AbsDAL
from .stored import ObservingBlock
from .stored import StoredProduct, StoredParameter


def tags_are_valid(subset, superset):
    for key, val in subset.items():
        if key in superset and superset[key] != val:
            return False
    return True


class BaseDictDAL(AbsDAL):
    def __init__(self, drps, ob_table, prod_table, req_table, extra_data=None):
        super(BaseDictDAL, self).__init__()

        # Check that the structure de base is correct
        self.ob_table = ob_table
        self.prod_table = prod_table
        self.req_table = req_table
        self.extra_data = extra_data if extra_data else {}
        self.drps = drps

    def search_instrument_configuration_from_ob(self, ob):
        ins = ob.instrument
        name = ob.configuration
        return self.search_instrument_configuration(ins, name)

    def search_instrument_configuration(self, ins, name):

        drp = self.drps.query_by_name(ins)

        if drp is None:
            raise NoResultFound('Instrument "{}" not found'.format(ins))
        try:
            this_configuration = drp.configurations[name]
        except KeyError:
            raise NoResultFound('Instrument configuration "{}" missing'.format(name))

        return this_configuration

    def search_oblock_from_id(self, obsid):
        try:
            ob = self.ob_table[obsid]
            return ObservingBlock(**ob)
        except KeyError:
            raise NoResultFound("oblock with id %d not found" % obsid)

    def search_recipe(self, ins, mode, pipeline):

        drp = self.drps.query_by_name(ins)

        if drp is None:
            raise NoResultFound('DRP not found')

        try:
            this_pipeline = drp.pipelines[pipeline]
        except KeyError:
            raise NoResultFound('pipeline not found')

        try:
            recipe = this_pipeline.get_recipe_object(mode)
            return recipe
        except KeyError:
            raise NoResultFound('mode not found')


    def search_recipe_fqn(self, ins, mode, pipename):

        drp = self.drps.query_by_name(ins)

        if drp is None:
            raise NoResultFound('result not found')

        try:
            this_pipeline = drp.pipelines[pipename]
        except KeyError:
            raise NoResultFound('result not found')

        recipes = this_pipeline.recipes
        try:
            recipe_fqn = recipes[mode]
        except KeyError:
            raise NoResultFound('result not found')

        return recipe_fqn

    def search_recipe_from_ob(self, ob):
        ins = ob.instrument
        mode = ob.mode
        pipeline = ob.pipeline
        return self.search_recipe(ins, mode, pipeline)

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

        klass = tipo.__class__
        drp = self.drps.query_by_name(ins)
        label = drp.product_label(klass)

        # search results of these OBs
        for prod in self.prod_table[ins]:
            pk = prod['type'] 
            pt = prod['tags']
            if pk == label and tags_are_valid(pt, tags):
                # this is a valid product
                # We have found the result, no more checks
                # Make a copy
                rprod = dict(prod)
                print(tipo, prod['content'])
                rprod['content'] = load(tipo, prod['content'])
                return StoredProduct(**rprod)
        else:
            msg = 'type %s compatible with tags %r not found' % (klass, tags)
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

    def search_product(self, name, tipo, obsres):
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

    def search_parameter(self, name, tipo, obsres):
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
                    value = load(name, prod['content'])
                    content = StoredParameter(value)
                    return content
            else:
                msg = 'name %s compatible with tags %r not found' % (name, tags)
                raise NoResultFound(msg)


class DictDAL(BaseDictDAL):
    def __init__(self, drps, base):

        # Check that the structure of 'base' is correct
        super(DictDAL, self).__init__(
            drps,
            base['oblocks'],
            base['products'],
            base['parameters']
        )
