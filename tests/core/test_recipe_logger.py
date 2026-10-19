"""The logger of the recipes"""

import logging

from numina.core import BaseRecipe

from tests.support.recipes import BiasRecipe


class ModuleRecipe(BaseRecipe):
    pass


class OwnLoggerRecipe(BaseRecipe):
    logger = logging.getLogger("instrument.recipes")


class InheritedLoggerRecipe(OwnLoggerRecipe):
    pass


class DerivedFromOtherModule(BiasRecipe):
    pass


def test_logger_of_the_module():
    """Without a logger in the class, the logger of its module"""
    assert ModuleRecipe.logger.name == __name__
    assert BiasRecipe.logger.name == "tests.support.recipes"
    assert DerivedFromOtherModule.logger.name == __name__


def test_logger_defined_in_the_recipe():
    """The logger defined in a recipe is used by its subclasses"""
    assert OwnLoggerRecipe.logger.name == "instrument.recipes"
    assert InheritedLoggerRecipe.logger.name == "instrument.recipes"


def test_logger_of_base_recipe():
    assert BaseRecipe.logger.name == "numina.recipes.numina"
