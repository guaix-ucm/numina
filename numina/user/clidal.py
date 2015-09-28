
# Copyright 2014 Universidad Complutense de Madrid
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

from numina.core.dal import AbsDAL
from numina.core.dal import NoResultFound
from numina.core.dal import StoredProduct
from numina.core.dal import StoredParameter
from numina.core.dal import ObservingBlock
from numina.core import init_drp_system
from numina.store import init_store_backends
from numina.core import import_object
from numina.core import obsres_from_dict

_logger = logging.getLogger("numina.simpledal")

def process_format_version_0(loaded_obs, loaded_data):
    return ComandLineDAL(loaded_obs, loaded_data)


class ComandLineDAL(AbsDAL):
    '''A DAL to use with the command line interface'''
    def __init__(self, ob_table, reqs):
        self.args_drps = init_drp_system()
        self.ob_table = ob_table
        init_store_backends()
        self._reqs = reqs

    def search_rib_from_ob(self, obsres, pipeline):
        return None

    def obsres_from_oblock_id(self, obsid):
        este = self.ob_table[obsid]
        obsres = obsres_from_dict(este)

        this_drp = self.args_drps[obsres.instrument]
        tagger = None
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
        return obsres

    def search_oblock_from_id(self, obsid):
        try:
            ob = self.ob_table[obsid]
            return ObservingBlock(**ob)
        except KeyError:
            raise NoResultFound("oblock with id %d not found", obsid)

    def search_recipe_from_ob(self, obsres, pipeline):

        _logger.info("Identifier of the observation result: %d", obsres.id)

        _logger.info("instrument name: %s", obsres.instrument)
        my_ins = self.args_drps.get(obsres.instrument)

        if my_ins is None:
            raise ValueError('no instrument named %r'% obsres.instrument)

        my_pipe = my_ins.pipelines.get(pipeline)

        if my_pipe is None:
            raise ValueError('no pipeline named %r'% pipeline)

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
        '''Returns the first coincidence...'''
        _logger.debug('search for instrument %s, req %s with tags %s',
                      ins, req, tags)
        key = req.dest
        try:
            content = self._reqs['requirements'][key]
        except KeyError:
            raise NoResultFound("key %s not found", key)

        return StoredProduct(id=-1, content=content, tags={})

    def search_prod_obsid(self, ins, obsid, pipeline):
        return StoredProduct(id=-1, content='null.fits', tags={})

    def search_param_req(self, req, instrument, mode, pipeline):
        key = req.dest
        try:
            param = self._reqs['requirements'][key]
            content = StoredParameter(param)
        except KeyError:
            raise NoResultFound("key %s not found", key)
        return content
