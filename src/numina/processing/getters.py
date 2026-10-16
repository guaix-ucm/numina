#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Getters of correctors, from the calibrations of the input of a recipe

A getter is a function ``getter(rinput, meta, ins, datamodel)`` that
returns a corrector, used by ``BaseRecipe.create_getters`` to build the
reduction flows of a recipe.
"""

import logging

import numina.util.node as node

_logger = logging.getLogger(__name__)


def load_corrector(rinput, key, corrector_class, datamodel, **kwargs):
    """The corrector of the calibration ``rinput.<key>``

    The corrector is built with the data of the primary HDU of the
    calibration, and its identifier, given by ``datamodel.get_imgid``.

    Parameters
    ----------
    rinput : RecipeInput
    key : str
        Name of the requirement with the calibration.
    corrector_class : type
        Class of the corrector, called as
        ``corrector_class(data, datamodel=datamodel, calibid=calibid, **kwargs)``.
    datamodel : DataModel
    kwargs
        Other arguments of the corrector.

    Returns
    -------
    Node
        The corrector, or a node that does nothing if the calibration
        is not given.
    """
    value = getattr(rinput, key, None)
    if value is None:
        _logger.info('"%s" not provided, ignored', key)
        return node.IdNode()
    with value.open() as hdul:
        calibid = datamodel.get_imgid(hdul)
        _logger.info('loading "%s", %s', key, calibid)
        return corrector_class(hdul[0].data, datamodel=datamodel, calibid=calibid, **kwargs)


def get_corrector_for(key, corrector_class, **kwargs):
    """A getter of the corrector of the calibration ``rinput.<key>``

    The getter calls :func:`load_corrector` with `key`, `corrector_class`
    and `kwargs`.
    """

    def getter(rinput, meta, ins, datamodel):
        return load_corrector(rinput, key, corrector_class, datamodel, **kwargs)

    getter.__name__ = f"get_corrector_{key}"
    getter.__doc__ = f"Corrector of the calibration {key}, with {corrector_class.__name__}"
    return getter
