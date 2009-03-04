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

import pyfits
import scipy.stsci.image as im

from emir.numina import RecipeBase
import emir.image.combine as com

logger = logging.getLogger("emir.recipes")
        
class BiasImaging(RecipeBase):
    pass

class DarkImaging(RecipeBase):
    '''Recipe for processing dark current images
    
    Here starts the long description...
    It continues several lines'''
    def __init__(self):
        super(DarkImaging, self).__init__(optusage = "usage: %prog [options] recipe [recipe-options]")
        # Default values. This can be read from a file
        self.iniconfig.add_section('inputs')
        self.iniconfig.set('inputs','files','')
                
    def process(self):
        pfiles = self.iniconfig.get('inputs','files')
        pfiles = ' '.join(pfiles.splitlines()).replace(',',' ').split()
        if len(pfiles) == 0:
            logger.warning('No files to process')
            return
        
        images = []
        try:
            for i in pfiles:
                logger.debug('Loading %s',i)
                images.append(pyfits.open(i))
        except IOError, err:
            logger.error(err)
            logger.debug('Cleaning up hdus')
            for i in images:
                i.close()
        
        logger.debug('We have %d images',len(images))
        # Data from the primary extension
        data = [i['primary'].data for i in images]
        #
        result = im.average(data)
        # Creating the result pyfits structure
        # Creating the primary HDU
        dhdu = pyfits.PrimaryHDU(result)
        # Variance and exposure extensions
        vhdu = pyfits.ImageHDU(name='VARIANCE')
        nhdu = pyfits.ImageHDU(name='NUMBER')
        # Final structure
        hdulist = pyfits.HDUList([dhdu, vhdu, nhdu])
        return hdulist
        
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
    '''Simulates EMIR images
    
    Here starts the long description...
    It continues several lines'''
    def __init__(self):
        super(SimulateImage, self).__init__(optusage = "usage: %prog [options] recipe [recipe-options]")
        # Default values. This can be read from a file
        self.iniconfig.add_section('readout')
        self.iniconfig.set('readout','mode','cds')
        self.iniconfig.set('readout','reads','3')
        self.iniconfig.set('readout','repeat','1')
        self.iniconfig.set('readout','scheme','perline')
        self.iniconfig.set('readout','exposure','0')
        self.iniconfig.add_section('detector')
        self.iniconfig.set('detector','shape','(2048,2048)')
        self.iniconfig.set('detector','ron','2.16')
        self.iniconfig.set('detector','dark','0.37')
        self.iniconfig.set('detector','gain','3.028')
        self.iniconfig.set('detector','flat','1')
        self.iniconfig.set('detector','well','65536')
        
    def setup(self):
        detector_conf = {}
        detector_conf['shape'] = eval(self.iniconfig.get('detector', 'shape'))
        for i in ['ron','dark','gain','flat', 'well']:
            detector_conf[i] = self.iniconfig.getfloat('detector', i)

        self.detector = EmirDetector(**detector_conf)
        logger.info('Created detector')
    
        readout_opt = {}
                
        for i in ['exposure']:
            readout_opt[i] = self.iniconfig.getfloat('readout', i)
        for i in ['reads', 'repeat']:
            readout_opt[i] = self.iniconfig.getint('readout', i)
        for i in ['mode', 'scheme']:
            readout_opt[i] = self.iniconfig.get('readout', i)
        
        self._repeat = readout_opt['repeat']
        
        logger.info('Detector configured')
        self.detector.configure(readout_opt)
        
        self.input = numpy.zeros(detector_conf['shape'])
        self.detector.exposure(readout_opt['exposure'])
        
        logger.info('FITS builder created')
        self.storage = Storage(Config.default_fits_headers)
        
        
    def process(self):
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


