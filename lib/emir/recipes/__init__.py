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

'''Recipes for Emir Observing Modes'''


import logging

import pyfits
import scipy.stsci.image as im

import emir.image.combine as com
from emir.recipes.darkimaging import DarkImaging
from emir.recipes.simulateimage import SimulateImage
from emir.numina import RecipeBase


__version__ = "$Id$"

_logger = logging.getLogger("emir.recipes")
        
class BiasImaging(RecipeBase):
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


class SlitCheck(RecipeBase):
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


class OffsetSpectra(RecipeBase):
    pass
        
if __name__ == "__main__":
    a = DitheredImage()
    print type(a.parser)
    b = RecipeBase()
    print type(b)


