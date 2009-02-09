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
    def __init__(self):
        self.parser = OptionParser(usage = "usage: %prog [options] recipe [recipe-options]")
    def run(self):
        pass
        
        
class BiasImaging(RecipeBase):
    pass

class DarkImaging(RecipeBase):
    pass

class IntensityFlatField(RecipeBase):
    pass

class SpectralFlatField(RecipeBase):
    pass

class SlitTransmissionCalibration(RecipeBase):
    pass 

class WavelengthCalibration(RecipeBase):
    pass 

class TsRoughFocs(RecipeBase):
    pass
        
class TsFineFocs(RecipeBase):
    pass

class EmirFocusControl(RecipeBase):
    pass

class TargetAcquisition(RecipeBase):
    pass

class MaskImage(RecipeBase):
    pass

class SlitCheck(Recipebase):
    pass
        
class StareImage(RecipeBase):
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

class NBImage(RecipeBase):
    pass

class DitheredImage(RecipeBase):
    pass
        
class MicroDitheredImage(RecipeBase):
    pass

class MosaicedImage(RecipeBase):
    pass

class StareSpectra(RecipeBase):
    pass

class DNSpectra(RecipeBase):
    pass

class OffsetSpectra(Recipebase):
    pass
        
class DirectImagingRecipe(RecipeBase):
    pass

if __name__ == "__main__":
    a = DirectImagingRecipe()
    print a.parser


