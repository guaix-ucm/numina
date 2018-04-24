#
# Copyright 2014-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DAL Mock class"""


from numina.core.utils import AlwaysSuccessRecipe
from .stored import StoredProduct, StoredParameter
from .stored import ObservingBlock
from .absdal import AbsDAL


class MockDAL(AbsDAL):

    def search_oblock_from_id(self, objid):
        ob = ObservingBlock(1, 'TEST_INSTRUMENT', 'TEST_MODE', [], [], None)
        return ob

    # Implemented in base class
    # def obsres_from_oblock_id(self, obsid):

    def search_recipe_from_ob(self, ob, pipeline='default'):

        return AlwaysSuccessRecipe

    def search_prod_type_tags(self, ins, type, tags, pipeline):
        '''Returns the first coincidence...'''
        return StoredProduct(id=100, content='null.fits', tags={})

    def search_prod_req_tags(self, req, ins, tags, pipeline):
        return self.search_prod_type_tags(req.type, ins, tags, pipeline)

    # Implemented in base class
    # def obsres_from_proc_oblock_id(self, instrument, child_id)

    def search_prod_obsid(self, ins, obsid, pipeline):
        return StoredProduct(id=100, content='null.fits',
                             tags={'readmode': 'a'})

    def search_param_req_tags(self, req, instrument, mode, tags, pipeline):
        return StoredParameter(content='parameter')
