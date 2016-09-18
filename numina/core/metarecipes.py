#
# Copyright 2008-2015 Universidad Complutense de Madrid
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

"""Metaclasses for Recipes."""

from .recipeinout import RecipeResult, RecipeInput
from .dataholders import Product
from .requirements import Requirement

_RECIPE_RESULT_NAME = 'RecipeResult'
_RECIPE_INPUT_NAME = 'RecipeInput'

class RecipeType(type):
    """Metaclass for Recipe."""
    def __new__(cls, classname, parents, attributes):

        filter_reqs = {}
        filter_prods = {}
        filter_attr = {}

        for name, val in attributes.items():
            if isinstance(val, Requirement):
                filter_reqs[name] = val
            elif isinstance(val, Product):
                filter_prods[name] = val
            else:
                filter_attr[name] = val

        BaseRecipeResult = cls.get_base_class(RecipeResult, parents, attributes, _RECIPE_RESULT_NAME)
        BaseRecipeInput = cls.get_base_class(RecipeInput, parents, attributes, _RECIPE_INPUT_NAME)

        ReqsClass = cls.create_inpt_class(classname, BaseRecipeInput, filter_reqs)
        ResultClass = cls.create_prod_class(classname, BaseRecipeResult, filter_prods)

        filter_attr[_RECIPE_RESULT_NAME] = ResultClass
        filter_attr[_RECIPE_INPUT_NAME] = ReqsClass

        return super(RecipeType, cls).__new__(
            cls, classname, parents, filter_attr)

    @classmethod
    def create_gen_class(cls, classname, baseclass, attributes):
        if attributes:
            klass = type(classname, (baseclass,), attributes)
        else:
            klass = baseclass
        return klass

    @classmethod
    def get_base_class(cls, default, parents, attributes, name):
        # Find base class for RecipeResult in parents
        if name in attributes:
            return attributes[name]
        else:
            for parent in parents:
                if hasattr(parent, name):
                    return getattr(parent, name)
            else:
                return default

    @classmethod
    def create_inpt_class(cls, classname, base, attributes):
        return cls.create_gen_class('%sInput' % classname,
                                    base, attributes)

    @classmethod
    def create_prod_class(cls, classname, base, attributes):
        return cls.create_gen_class('%sResult' % classname,
                                    base, attributes)
