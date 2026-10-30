#
# Copyright 2014-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DAL base class"""

from abc import ABCMeta, abstractmethod


class DALInterface(metaclass=ABCMeta):
    """Abstract Base Class for DAL"""

    @abstractmethod
    def obsres_from_oblock_id(self, obsid):
        """Return the observation result of the observing block with id `obsid`."""
        # This implementation does not depend
        # on the details of the DAL...
        pass

    @abstractmethod
    def search_recipe_from_ob(self, ob, pipeline="default"):
        """Return the recipe of the mode of the observing block `ob`."""
        # returns RecipeClass
        pass

    @abstractmethod
    def search_product(self, name, tipo, obsres, options=None):
        """Return the product `name` of type `tipo` for `obsres`, a ``StoredProduct``."""
        # returns StoredProduct
        pass

    @abstractmethod
    def search_parameter(self, name, tipo, obsres, options=None):
        """Return the parameter `name` of type `tipo` for `obsres`, a ``StoredParameter``."""
        # returns StoredProduct
        pass

    @abstractmethod
    def search_result_relative(self, name, tipo, obsres, result_desc, options=None):
        """Return the value of a result of other observing blocks, described by `result_desc`."""
        pass

    def has_extra(self, name):
        """True if a value for requirement `name` was given in the command line"""
        return False

    def update_configuration(self, obsres):
        """Update the instrument configuration, after adding frames to obsres"""
        pass
