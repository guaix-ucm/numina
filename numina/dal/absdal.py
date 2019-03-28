#
# Copyright 2014-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DAL base classes"""


from .daliface import DALInterface
from numina.exceptions import NoResultFound


class AbsDAL(DALInterface):
    pass


class AbsDrpDAL(DALInterface):
    def __init__(self, drps, *args, **kwargs):
        super(AbsDrpDAL, self).__init__()
        self.drps = drps

    def search_instrument_configuration_from_ob(self, obsres):
        from numina.instrument.assembly import assembly_instrument
        this_drp = self.drps.query_by_name(obsres.instrument)
        key, date_obs, keyname = this_drp.select_profile(obsres)
        ins = assembly_instrument(this_drp.configurations, key, date_obs, by_key=keyname)
        return ins

    def search_instrument_configuration(self, keyval, value, by_key='name'):
        from numina.instrument.assembly import assembly_instrument
        drp = self.drps.query_by_name(keyval)
        ins = assembly_instrument(drp.configurations, keyval, value, by_key=by_key)
        return ins

    def search_recipe(self, ins, mode, pipeline):
        drp = self.drps.query_by_name(ins)
        return drp.get_recipe_object(mode, pipeline_name=pipeline)

    def search_recipe_fqn(self, ins, mode, pipename):

        drp = self.drps.query_by_name(ins)

        this_pipeline = drp.pipelines[pipename]
        recipes = this_pipeline.recipes
        recipe_fqn = recipes[mode]
        return recipe_fqn

    def search_recipe_from_ob(self, ob, pipeline='default'):
        instrument = ob.instrument
        mode = ob.mode
        pipeline = ob.pipeline
        return self.search_recipe(instrument, mode, pipeline)
