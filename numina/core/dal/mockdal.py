#
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

'''DAL Mock class'''


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

    def search_recipe_from_ob(self, ob, pipeline):

        return AlwaysSuccessRecipe

    def search_prod_type_tags(self, ins, type, tags, pipeline):
        '''Returns the first coincidence...'''
        return StoredProduct(id=100, content='null.fits', tags={})

    def search_prod_req_tags(self, req, ins, tags, pipeline):
        return self.search_prod_type_tags(req.type, ins, tags, pipeline)

    # Implemented in base class
    # def obsres_from_proc_oblock_id(self, instrument, child_id)

    def search_prod_obsid(self, ins, obsid):
        return StoredProduct(id=100, content='null.fits',
                             tags={'readmode': 'a'})

    def search_param_req(self, req, instrument, mode, pipeline):
        return StoredParameter(content='parameter')
