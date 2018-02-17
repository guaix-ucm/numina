# Copyright 2014-2017 Universidad Complutense de Madrid
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

import logging

from numina.exceptions import NoResultFound
from numina.dal.absdal import AbsDrpDAL
from numina.dal import StoredProduct
from numina.dal import StoredParameter
from numina.dal import ObservingBlock
from numina.core import import_object
from numina.core import obsres_from_dict
import numina.store as storage


_logger = logging.getLogger("numina.dal.clidal")


class CommandLineDAL(AbsDrpDAL):
    """A DAL to use with the command line interface"""

    def __init__(self, drps, ob_table, reqs, extra_reqs=None):
        super(CommandLineDAL, self).__init__(drps)
        self.ob_table = ob_table
        self._reqs = reqs
        if extra_reqs:
            self._reqs['requirements'].update(extra_reqs)

    def search_rib_from_ob(self, obsres, pipeline):
        return None

    def obsres_from_oblock_id(self, obsid, configuration=None):
        este = self.ob_table[obsid]
        obsres = obsres_from_dict(este)

        this_drp = self.drps.query_by_name(obsres.instrument)
        tagger = None
        for mode in this_drp.modes:
            if mode.key == obsres.mode:
                tagger = mode.tagger
                break
        else:
            raise ValueError(
                'no mode for {0}.mode in instrument {0}.instrument'.format(
                    obsres))

        if tagger is None:
            master_tags = {}
        else:
            master_tags = tagger(obsres)

        obsres.tags = master_tags

        # Insert Instrument configuration
        if configuration:
            # override instrument configuration
            obsres.configuration = self.search_instrument_configuration(obsres.instrument, configuration)
        else:
            # Insert Instrument configuration
            obsres.configuration = this_drp.configuration_selector(obsres)

        return obsres

    def search_oblock_from_id(self, obsid):
        try:
            ob = self.ob_table[obsid]
            return ObservingBlock(**ob)
        except KeyError:
            raise NoResultFound("oblock with id %s not found" % obsid)

    def search_recipe_from_ob(self, obsres, pipeline='default'):

        _logger.info("Identifier of the observation result: %s", obsres.id)

        _logger.info("instrument name: %s", obsres.instrument)
        my_ins = self.drps.query_by_name(obsres.instrument)
        if my_ins is None:
            raise ValueError('no instrument named %r' % obsres.instrument)

        pipeline = obsres.pipeline
        my_pipe = my_ins.pipelines.get(pipeline)

        if my_pipe is None:
            raise ValueError('no pipeline named %r' % pipeline)

        _logger.info("observing mode: %r", obsres.mode)

        recipe_fqn = my_pipe.recipes.get(obsres.mode)
        recipeclass = import_object(recipe_fqn)

        return recipeclass

    def search_prod_type_tags(self, typo, ins, tags, pipeline):
        '''Returns the first coincidence...'''
        _logger.debug('search for instrument %s, type %s with tags %s',
                      ins, typo, tags)
        return StoredProduct(id=100, content='null.fits', tags={})

    def search_prod_req_tags(self, req, ins, tags, pipeline):
        """Returns the first coincidence..."""
        _logger.debug('search for instrument %s, req %s with tags %s',
                      ins, req, tags)
        key = req.dest
        try:
            product = self._reqs['requirements'][key]
            content = storage.load(req.type, product)
        except KeyError:
            raise NoResultFound("key %s not found" % key)

        return StoredProduct(id=-1, content=content, tags={})

    def search_prod_obsid(self, ins, obsid, pipeline):
        return StoredProduct(id=-1, content='null.fits', tags={})

    def search_param_req(self, req, instrument, mode, pipeline):
        key = req.dest
        _logger.debug('search for instrument %s, req %s',
                      instrument, req)
        try:
            param = self._reqs['requirements'][key]
            content = storage.load(req.type, param)

        except KeyError:
            raise NoResultFound("key %s not found" % key)
        return StoredParameter(content)

    def search_param_req_tags(self, req, instrument, mode, tags, pipeline):
        raise NotImplementedError

    def search_result_relative(self, name, obsres, mode, field, node, options=None):
        # mode field node could go together...
        return []


