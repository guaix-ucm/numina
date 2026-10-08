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
    pass


class AbsDrpDAL(DALInterface):
    def __init__(self, drps, *args, **kwargs):
        super().__init__()
        self.drps = drps

    def search_recipe(self, ins, mode, pipeline):
        drp = self.drps.query_by_name(ins)
        return drp.get_recipe_object(mode, pipeline_name=pipeline)

    def search_recipe_from_ob(self, ob, pipeline="default"):
        instrument = ob.instrument
        mode = ob.mode
        pipeline = ob.pipeline
        return self.search_recipe(instrument, mode, pipeline)
