#
# Copyright 2014-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DAL base class"""

from abc import ABCMeta, abstractmethod


class DALInterface(metaclass=ABCMeta):
    """Abstract Base Class for DAL"""

    @abstractmethod
    def obsres_from_oblock_id(self, obsid):
        # This implementation does not depend
        # on the details of the DAL...
        pass

    @abstractmethod
    def search_recipe_from_ob(self, ob, pipeline='default'):
        # returns RecipeClass
        pass

    @abstractmethod
    def search_product(self, name, tipo, obsres, options=None):
        # returns StoredProduct
        pass

    @abstractmethod
    def search_parameter(self, name, tipo, obsres, options=None):
        # returns StoredProduct
        pass

    @abstractmethod
    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):
        pass
