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
from tempfile import mkstemp, mkdtemp
import shutil
import os

import yaml
import numpy as np
from astropy.io import fits

from numina.core import init_drp_system, import_object
from numina.core.oresult import ObservationResult
from numina.core.reciperesult import ErrorRecipeResult
from numina.core.recipeinput import RecipeInputBuilder
from numina.core import obsres_from_dict
from numina.user import main

import numina.tests.drps as namespace

class UserTestCase(unittest.TestCase):
    '''Test of the user CLI.'''

    def setUp(self):
        self.workdir = mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.workdir)
        
        
    def test0(self):
        # create fits
        
        somefits = []
        nimg = 10
        for _ in range(nimg):
            hdu = fits.PrimaryHDU(data=np.zeros((10,10), dtype='int16'))
            hdul = fits.HDUList([hdu])
            _, filename = mkstemp(dir=self.workdir, suffix='.fits')
            hdul.writeto(filename, clobber=True)
            somefits.append(filename)

        #
        # Create the recipe_input
        obsres = ObservationResult()
        obsres.frames = somefits
        
        _, obsresfile = mkstemp(dir=self.workdir)
        yaml.dump(obsres.__dict__, open(obsresfile, 'wb'))
        # create obs res
        
        val = main(['run-recipe', 'emir.recipes.aiv.SimpleBiasRecipe', 
                    '--obs-res', 'dum1.txt', '--basedir', self.workdir])
        
        
        print os.listdir(self.workdir + '/_results')
        print open(self.workdir + '/_results/result.txt').read()
        self.assertEqual(val, "dum1")

    def otest1(self):
        
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

    def otest2(self):
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

    def otest3(self):
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


