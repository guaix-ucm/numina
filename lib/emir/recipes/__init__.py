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

from numina import RecipeBase
import emir.image.combine as com
from darkimaging import DarkImaging
from simulateimage import SimulateImage


__version__ = "$Revision$"

_logger = logging.getLogger("emir.recipes")
        
class BiasImaging(RecipeBase):
    '''Recipe to process data taken in Bias Image Mode.'''
    pass

        
class IntensityFlatField(RecipeBase):
    '''Recipe to process data taken in Intensity Flat Field Mode.'''
    pass


class SpectralFlatField(RecipeBase):
    '''Recipe to process data taken in Spectral Flat Field Mode.'''
    pass


class SlitTransmissionCalibration(RecipeBase):
    '''Recipe to process data taken in Slit transmission calibration Mode.'''
    pass 


class WavelengthCalibration(RecipeBase):
    '''Recipe to process data taken in wavelength calibration Mode.'''
    pass 


class TsRoughFocs(RecipeBase):
    '''Recipe to process data taken in Telescope rough focus Mode.'''
    pass

        
class TsFineFocs(RecipeBase):
    '''Recipe to process data taken in Telescope fine focus Mode.'''
    pass


class EmirFocusControl(RecipeBase):
    '''Recipe to process data taken in Emir focus control Mode.'''
    pass


class TargetAcquisition(RecipeBase):
    '''Recipe to process data taken in Target acquisition Mode.'''
    pass


class MaskImage(RecipeBase):
    '''Recipe to process data taken in Mask image Mode.'''
    pass


class SlitCheck(RecipeBase):
    '''Recipe to process data taken in Slit check Mode.'''
    pass

        
class StareImage(RecipeBase):
    '''Recipe to process data taken in Stare image Mode.
    
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
    '''Recipe to process data taken in Nodded/Beamswitched image Mode.'''
    pass


class DitheredImage(RecipeBase):
    '''Recipe to process data taken in Dithered image Mode.'''
    pass

        
class MicroDitheredImage(RecipeBase):
    '''Recipe to process data taken in Microdithered image Mode.'''
    pass


class MosaicedImage(RecipeBase):
    '''Recipe to process data taken in Mosaiced image Mode.'''
    pass


class StareSpectra(RecipeBase):
    '''Recipe to process data taken in Stare spectra Mode.'''
    pass


class DNSpectra(RecipeBase):
    '''Recipe to process data taken in Dithered/Nodded spectra Mode.'''
    pass


class OffsetSpectra(RecipeBase):
    '''Recipe to process data taken in Offset spectra Mode.'''
    pass

