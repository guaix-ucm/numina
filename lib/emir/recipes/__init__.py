#!/usr/bin/env python

#
# Copyright 2008-2009 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$

import logging
from optparse import OptionParser

logger = logging.getLogger("emir.recipes")

class RecipeBase:
    '''Base class for Recipes of all kinds'''
    def run(self):
        pass


class Recipe(RecipeBase):
    '''Short description of the recipe.
    
    Long description of the recipe.
    Its spans several lines'''
    
    parser =  OptionParser(usage = "usage: %prog [options] recipe [recipe-options]")
    parser.add_option('-e',action="store_true", dest="test", default=False, help="test documentation")
    
    def run(self):
        logger.info("Hello, I\'m Recipe")
        
        
        
class DirectImagingRecipe(RecipeBase):
    '''Recipe to process data taken in Direct Imaging Mode.
    
    Inputs of the recipe:
        Science frames
        Offsets between them
        Master Dark
        Bad pixel mask (BPM)
        Non-linearity correction polynomials
        Master flat
        Master background
        Exposure Time (must be the same in all the frames)
        Airmass for each frame
        Detector model (gain, RN)
        Average extinction in the filter
    '''
    pass  

