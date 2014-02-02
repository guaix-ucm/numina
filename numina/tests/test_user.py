#
# Copyright 2008-2014 Universidad Complutense de Madrid
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
 
'''Unit test for CLI.'''

import unittest
from numina.core import init_drp_system, import_object
from numina.core import RequirementParser, obsres_from_dict
from numina.core.reciperesult import ErrorRecipeResult
from numina.core.recipeinput import RecipeInputBuilder

class UserTestCase(unittest.TestCase):
    '''Test of the user CLI.'''

    def test1(self):
        import numina.tests.drps as namespace
        drps = init_drp_system(namespace)
        obsres = obsres_from_dict({'mode': 'success', 
            'instrument': 'CLODIA', 'frames': [], 'configuration': 'default',
            'pipelines': 'default'})
        ins_name = obsres.instrument
        my_ins = drps.get(ins_name)
        
        my_ins_conf = my_ins.configurations.get(obsres.configuration)
        my_pipe = my_ins.pipelines.get(obsres.pipeline)
        obs_mode = obsres.mode
        recipe_fqn = MyRecipeClass = my_pipe.recipes.get(obs_mode)
    
        MyRecipeClass = import_object(recipe_fqn)
        
        rib = RecipeInputBuilder()
        ri = rib.build(MyRecipeClass, obsres, {})
        recipe = MyRecipeClass()
        recipe.configure(instrument=my_ins)
        #
        result = recipe(ri)
        self.assertIsInstance(result, MyRecipeClass.RecipeResult)

    def test2(self):
        import numina.tests.drps as namespace
        drps = init_drp_system(namespace)
        obsres = obsres_from_dict({'mode': 'fail', 
            'instrument': 'CLODIA', 'frames': [], 'configuration': 'default',
            'pipelines': 'default'})
        ins_name = obsres.instrument
        my_ins = drps.get(ins_name)
        
        my_ins_conf = my_ins.configurations.get(obsres.configuration)
        my_pipe = my_ins.pipelines.get(obsres.pipeline)
        obs_mode = obsres.mode
        recipe_fqn = MyRecipeClass = my_pipe.recipes.get(obs_mode)
    
        MyRecipeClass = import_object(recipe_fqn)
        
        rib = RecipeInputBuilder()
        ri = rib.build(MyRecipeClass, obsres, {})
        recipe = MyRecipeClass()
        recipe.configure(instrument=my_ins)
        #
        result = recipe(ri)
        self.assertIsInstance(result, ErrorRecipeResult)

    def test3(self):
        import numina.tests.drps as namespace
        drps = init_drp_system(namespace)
        obsres = obsres_from_dict({'mode': 'success_obs', 
            'instrument': 'CLODIA', 'frames': [], 'configuration': 'default',
            'pipelines': 'default'})
        ins_name = obsres.instrument
        my_ins = drps.get(ins_name)
        
        my_ins_conf = my_ins.configurations.get(obsres.configuration)
        my_pipe = my_ins.pipelines.get(obsres.pipeline)
        obs_mode = obsres.mode
        recipe_fqn = MyRecipeClass = my_pipe.recipes.get(obs_mode)
    
        MyRecipeClass = import_object(recipe_fqn)
        
        rib = RecipeInputBuilder()
        ri = rib.build(MyRecipeClass, obsres, {})
        recipe = MyRecipeClass()
        recipe.configure(instrument=my_ins)
        #
        result = recipe(ri)
        self.assertIsInstance(result, ErrorRecipeResult)


