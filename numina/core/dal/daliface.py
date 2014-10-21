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

'''DAL base class'''

from abc import ABCMeta, abstractmethod


class NoResultFound(Exception):
    pass


class DALInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def search_oblock_from_id(self, objid):
        # returns ObservingBlock object
        pass

    @abstractmethod
    def obsres_from_oblock_id(self, obsid):
        # This implementation does not depend
        # on the details of the DAL...
        pass

    @abstractmethod
    def search_recipe_from_ob(self, ob, pipeline):
        # returns RecipeClass
        pass

    @abstractmethod
    def search_rib_from_ob(self, ob, pipeline):
        # returns RecipeInputBuilder
        pass

    @abstractmethod
    def obsres_from_proc_oblock_id(self, instrument, child_id):
        pass

    @abstractmethod
    def search_prod_obsid(self, instrument, obsid, pipeline):
        # returns StoredProduct
        pass

    @abstractmethod
    def search_prod_req_tags(self, req, instrument, tags, pipeline):
        # returns StoredProduct
        pass

    @abstractmethod
    def search_prod_type_tags(self, tipo, instrument, tags, pipeline):
        '''Returns the first coincidence...'''
        # returns StoredProduct
        pass

    @abstractmethod
    def search_param_req(self, req, instrument, mode, pipeline):
        # returns StoredParameter
        pass
