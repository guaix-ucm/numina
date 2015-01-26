
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
from numina.core import import_object

_logger = logging.getLogger("numina.simpledal")


class ComandLineDAL(AbsDAL):
    '''A DAL to use with the command line interface'''
    def __init__(self, reqs):
        self.args_drps = init_drp_system()
        self._reqs = reqs

    def search_oblock_from_id(self, objid):
        ob = ObservingBlock(-1, 'null', 'null', [], [], None)
        return ob

    def search_recipe_from_ob(self, ob):

        class Args():
            pass
        args = Args()
        args.drps = self.args_drps
        args.insconf = None
        args.pipe_name = 'default'
        res = load_from_obsres(ob, args)
        recipe_fqn, _pipe_name, _my_ins_conf, _ins_conf = res

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

    def search_prod_obsid(self, ins, obsid):
        return StoredProduct(id=-1, content='null.fits', tags={})

    def search_param_req(self, req, instrument, mode, pipeline):
        key = req.dest
        try:
            param = self._reqs['requirements'][key]
            content = StoredParameter(param)
        except KeyError:
            raise NoResultFound("key %s not found", key)
        return content

import sys
import yaml
from numina.core import InstrumentConfiguration


# FIXME: This function must not call exit at all
def load_from_obsres(obsres, args):
    _logger.info("Identifier of the observation result: %d", obsres.id)
    ins_name = obsres.instrument
    _logger.info("instrument name: %s", ins_name)
    my_ins = args.drps.get(ins_name)
    if my_ins is None:
        _logger.error('instrument %r does not exist', ins_name)
        sys.exit(1)

    _logger.debug('instrument is %s', my_ins)
    # Load configuration from the command line
    if args.insconf is not None:
        _logger.debug("configuration from CLI is %r", args.insconf)
        ins_conf = args.insconf
    else:
        ins_conf = obsres.configuration

    _logger.info('loading instrument configuration %r', ins_conf)
    my_ins_conf = my_ins.configurations.get(ins_conf)

    if my_ins_conf:
        _logger.debug('instrument configuration object is %r', my_ins_conf)
    else:
        # Trying to open a file
        try:
            with open(ins_conf) as fd:
                values = yaml.load(fd)
            if values is None:
                _logger.warning('%r is empty', ins_conf)
                values = {}
            else:
                # FIXME this file should be validated
                _logger.warning('loading no validated '
                                'instrument configuration')
                _logger.warning('you were warned')

            ins_conf = values.get('name', ins_conf)
            my_ins_conf = InstrumentConfiguration(ins_conf, values)

            # The new configuration must not overwrite existing configurations
            if ins_conf not in my_ins.configurations:
                my_ins.configurations[ins_conf] = my_ins_conf
            else:
                _logger.error('a configuration already '
                              'exists %r, exiting', ins_conf)
            sys.exit(1)

        except IOError:
            _logger.error('instrument configuration %r '
                          'does not exist', ins_conf)
            sys.exit(1)

    # Loading the pipeline
    if args.pipe_name is not None:
        _logger.debug("pipeline from CLI is %r", args.pipe_name)
        pipe_name = args.pipe_name
    else:
        pipe_name = obsres.pipeline
        _logger.debug("pipeline from ObsResult is %r", pipe_name)

    my_pipe = my_ins.pipelines.get(pipe_name)
    if my_pipe is None:
        _logger.error('instrument %r does not have '
                      'pipeline named %r', ins_name, pipe_name)
        sys.exit(1)

    _logger.info('loading pipeline %r', pipe_name)
    _logger.debug('pipeline object is %s', my_pipe)

    obs_mode = obsres.mode
    _logger.info("observing mode: %r", obs_mode)

    recipe_fqn = my_pipe.recipes.get(obs_mode)
    if recipe_fqn is None:
        _logger.error('pipeline %r does not have '
                      'recipe to process %r obs mode',
                      pipe_name, obs_mode)
        sys.exit(1)
    return recipe_fqn, pipe_name, my_ins_conf, ins_conf
