#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

from numina.core import RequirementParser
from numina.core.load import dict_requirement_lookup


class RecipeInputBuilder(object):

    def build(self, workenv, klass, mreqs):
        lc = dict_requirement_lookup(mreqs)
        rp = RequirementParser(klass, lookup=lc)
        requires = rp.parse()

        workenv.sane_work()
        workenv.copyfiles(mreqs['obresult'], requires)

        return requires

import logging

from numina.core import ObservationResult
from numina.core.dal import ObservingBlock
from numina.core.products import ObservationResultType
from numina.core.products import InstrumentConfigurationType
from numina.core import FrameDataProduct
from numina.core.dal import NoResultFound

_logger = logging.getLogger('numina.ri')


class RecipeInputBuilderGTC(object):
    '''Recipe Input Builder with GTC interface'''
    def __init__(self, recipeClass, dal):
        self.dal = dal
        self.recipeClass = recipeClass

    def buildRI(self, ob):
        RecipeRequirementsClass = self.recipeClass.RecipeRequirements

        result = {}
        pipeline = 'default'

        # We have to decide if the ob input
        # is a plain description (ObservingBlock)
        # or if it contains the nested results (Obsres)
        #
        # it has to contain the tags corresponding to the observing modes...
        if isinstance(ob, ObservingBlock):
            # We have to build an Obsres
            obsres = self.dal.obsres_from_oblock_id(ob.id)
        elif isinstance(ob, ObservationResult):
            # We have one
            obsres = ob
        else:
            raise ValueError('ob input is neither a ObservingBlock'
                             ' nor a ObservationResult')

        tags = getattr(obsres, 'tags', {})

        for key, req in RecipeRequirementsClass.items():

            if isinstance(req.type, ObservationResultType):
                result[key] = obsres
            elif isinstance(req.type, InstrumentConfigurationType):
                # Not sure how to handle this, or if it is needed...
                result[key] = {}
            elif isinstance(req.type, FrameDataProduct):
                try:
                    prod = self.dal.search_prod_req_tags(
                        req, obsres.instrument,
                        tags, pipeline
                        )

                    result[key] = prod.content
                except NoResultFound:
                    _logger.debug('No value found for %s', key)
            else:
                # Still not clear what to do with the other types
                try:
                    param = self.dal.search_param_req(
                        req, obsres.instrument,
                        obsres.mode, pipeline
                        )
                    result[key] = param.content
                except NoResultFound:
                    _logger.debug('No value found for %s', key)

        print result

        return RecipeRequirementsClass(**result)
