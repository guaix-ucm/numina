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

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Base class for Recipes of all kinds'''
    def __init__(self):
        self.parser = OptionParser(usage = "usage: %prog [options] recipe [recipe-options]")
    def setup(self):
        pass
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

from emir.simulation.detector import EmirDetector
from emir.simulation.storage import Storage
from emir.simulation.progconfig import Config
import numpy

class SimulateImage(RecipeBase):
    
    def setup(self):
        shape = (2048, 2048)    
        detector_conf = {'shape': shape, 'ron' : 2.16,
              'dark':0.37, 'gain': 3.028,
              'flat': 1, 'well': 2 ** 16 }

        self.detector = EmirDetector(**detector_conf)
        logger.info('Created detector')
    
        readout_opt = {'mode': 'cds' , 'reads': 3, 'repeat': 1, 'scheme': 'perline'}
        
        logger.info('Detector configured')
        self.detector.configure(readout_opt)
        
        self.input = numpy.zeros(shape)
        exposure = 10
        self.detector.exposure(exposure)
        
        logger.info('FITS builder created')
        self.storage = Storage(Config.default_fits_headers)
        
        
    def run(self):
        logger.info('Creating simulated array')    
        output = self.detector.path(self.input)
        
        header = {'RUN': '00001'}
        logger.info('Collecting metadata')
        header.update(self.detector.metadata())
        logger.info('Building pyfits image')
        hdulist = self.storage.store(output, headers=header)
        return hdulist
        
if __name__ == "__main__":
    a = DitheredImage()
    print type(a.parser)
    b = RecipeBase()
    print type(b)


