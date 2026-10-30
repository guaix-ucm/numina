#
# Copyright 2014-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DAL base classes"""

from .daliface import DALInterface
from numina.exceptions import NoResultFound  # noqa: F401


class AbsDAL(DALInterface):
    """Base class of the DALs, with the methods of ``DALInterface``."""

    pass


class AbsDrpDAL(DALInterface):
    """DAL that finds the recipes in the DRPs.

    Parameters
    ----------
    drps : DrpBase
        The DRPs, queried by instrument name.
    """

    def __init__(self, drps, *args, **kwargs):
        super().__init__()
        self.drps = drps

    def search_recipe(self, ins, mode, pipeline):
        """Return the recipe of `mode` in the pipeline `pipeline` of the DRP of instrument `ins`."""
        drp = self.drps.query_by_name(ins)
        return drp.get_recipe_object(mode, pipeline_name=pipeline)

    def search_recipe_from_ob(self, ob, pipeline="default"):
        """Return the recipe of the mode of `ob`, in the pipeline of `ob`."""
        instrument = ob.instrument
        mode = ob.mode
        pipeline = ob.pipeline
        return self.search_recipe(instrument, mode, pipeline)
